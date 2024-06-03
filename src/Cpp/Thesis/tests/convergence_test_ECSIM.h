#ifndef ECSIM_CONVERGENCE_TEST_H
#define ECSIM_CONVERGENCE_TEST_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

struct Options {
public:
    double dt;
    int Nx;
    int Nt;
    double dx;
    int Np;
    int Nsub = 1;
    double theta = 0.5;
    double qom = -1;
    double L;
    double t0 = 0;
};

void DecoupledSolve(ArrayXd xp, Array3Xd vp, Array3Xd E0, Array3Xd Bc, double t0, double t1, Eigen::Ref<ArrayXd> yn, Options* opt) {
    double m = 1;
    double q = -1;
    auto E = [opt,m,q](double t) {Array3d res(3); res << -m/q*sin(t) + m / (2 * q) * cos(t), -m / (2 * q) * sin(t), -m/ (2 *q) * sin(t) - 3 *m / (2 * q) * cos(t); return res; };
    auto B = [opt, m, q](double t) {Array3d res(3); res << -m / (2 * q), m / (2 * q), m/ (2 * q); return res; };

    std::vector<Eigen::Matrix3d> alphap(opt->Np); // column major
    Eigen::Array3Xd J0(Bc.rows(), Bc.cols());

    std::vector<Eigen::SparseMatrix<double>> Ms(9); // column major storage

    Eigen::SparseMatrix<double> AmpereX(opt->Nx, opt->Nx);
    Eigen::SparseMatrix<double> AmpereY(opt->Nx, opt->Nx);
    Eigen::SparseMatrix<double> AmpereZ(opt->Nx, opt->Nx);

    Eigen::SparseMatrix<double> Derv(opt->Nx, opt->Nx);
    Eigen::SparseMatrix<double> Derc(opt->Nx, opt->Nx);
    std::vector<Triplet<double>> tripletListDerc(opt->Nx * 2);
    std::vector<Triplet<double>> tripletListDerv(opt->Nx * 2);

    Eigen::SparseMatrix<double> NxIdentity(opt->Nx, opt->Nx);
    NxIdentity.setIdentity();
    Eigen::SparseMatrix<double> NxZeros(opt->Nx, opt->Nx);
    NxZeros.setZero();

    Eigen::VectorXd bKrylov(5 * opt->Nx);
    Eigen::VectorXd xKrylov = VectorXd::Zero(5 * opt->Nx);

    Eigen::SparseMatrix<double> Maxwell(5 * opt->Nx, 5 * opt->Nx);

    SparseLU<SparseMatrix<double>> solver;

    std::vector<Triplet<double>> tripletListMaxwell;

    double beta = opt->qom * opt->dt / 2;

    int Nx = opt->Nx;
    double dx = opt->dx;
    double t = t0;
    for (size_t step = 0; step < opt->Nt; step++) {
        auto tic = std::chrono::high_resolution_clock::now();
        //watch(B);

        J0.setZero();
        for (int ip = 0; ip < opt->Np; ip++) {
            auto Bp = B(t);
            double sx = Bp(0) * beta;
            double sy = Bp(1) * beta;
            double sz = Bp(2) * beta;

            alphap[ip].col(0) << 1 + sx * sx, -sz + sx * sy, sx* sz + sy;
            alphap[ip].col(1) << sz + sx * sy, 1 + sy * sy, -sx + sy * sz;
            alphap[ip].col(2) << sx * sz - sy, sx + sy * sz, 1 + sz * sz;
            alphap[ip].array() /= 1 + (Bp * (beta)).pow(2).sum();


        }
        // Setting Ms in columnmajor ordering -> [M0_0, M1_0, M2_0, M0_1, M1_1, M2_1, ...]
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
                Ms[3 * j + i].resize(opt->Nx, opt->Nx);
                Ms[3 * j + i].setZero();
            }
        }

        AmpereX = NxIdentity + opt->qom * opt->dt * opt->dt * opt->theta / 2 * Ms[0] / opt->dx;
        AmpereY = NxIdentity + opt->qom * opt->dt * opt->dt * opt->theta / 2 * Ms[4] / opt->dx;
        AmpereZ = NxIdentity + opt->qom * opt->dt * opt->dt * opt->theta / 2 * Ms[8] / opt->dx;

        for (int i = 0; i < opt->Nx; i++) {
            tripletListDerv[2 * i] = Triplet<double>(i, i, opt->dt * opt->theta / opt->dx );
            tripletListDerv[2 * i + 1] = Triplet<double>(i, (i - 1 + opt->Nx) % opt->Nx, -opt->dt * opt->theta / opt->dx );

            tripletListDerc[2 * i] = Triplet<double>(i, i, -opt->dt * opt->theta / opt->dx );
            tripletListDerc[2 * i + 1] = Triplet<double>(i, (i + 1) % opt->Nx, opt->dt * opt->theta / opt->dx );
        }
        Derv.setFromTriplets(tripletListDerv.begin(), tripletListDerv.end());
        Derc.setFromTriplets(tripletListDerc.begin(), tripletListDerc.end());

        bKrylov << (E0.row(0)).transpose(), (E0.row(1)).transpose(), (E0.row(2) ).transpose(), Bc.row(1).transpose(), Bc.row(2).transpose();
        tripletListMaxwell.resize(AmpereX.nonZeros() + AmpereY.nonZeros() + AmpereZ.nonZeros() + Ms[1].nonZeros() + Ms[2].nonZeros() + Ms[3].nonZeros() +
            Ms[5].nonZeros() + Ms[6].nonZeros() + Ms[7].nonZeros() + 2 * Derv.nonZeros() + 2 * Derc.nonZeros() + 2 * opt->Nx);

        int index = 0;
        // First Rowblock
        for (int k = 0; k < AmpereX.outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(AmpereX, k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row(), it.col(), it.value());
            }
        for (int k = 0; k < Ms[3].outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Ms[3], k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row(), it.col() + opt->Nx, opt->qom * opt->dt * opt->dt * opt->theta / 2 / opt->dx * it.value());
            }

        for (int k = 0; k < Ms[6].outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Ms[6], k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row(), it.col() + 2 * opt->Nx, opt->qom * opt->dt * opt->dt * opt->theta / 2 / opt->dx * it.value());
            }

        // Second Rowblock
        for (int k = 0; k < Ms[1].outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Ms[1], k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 1 * opt->Nx, it.col() + 0 * opt->Nx, opt->qom * opt->dt * opt->dt * opt->theta / 2 / opt->dx * it.value());
            }

        for (int k = 0; k < AmpereY.outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(AmpereY, k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 1 * opt->Nx, it.col() + 1 * opt->Nx, it.value());
            }
        for (int k = 0; k < Ms[7].outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Ms[7], k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 1 * opt->Nx, it.col() + 2 * opt->Nx, opt->qom * opt->dt * opt->dt * opt->theta / 2 / opt->dx * it.value());
            }
        for (int k = 0; k < Derv.outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Derv, k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 1 * opt->Nx, it.col() + 4 * opt->Nx, it.value());
            }

        // Third Rowblock
        for (int k = 0; k < Ms[2].outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Ms[2], k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 2 * opt->Nx, it.col() + 0 * opt->Nx, opt->qom * opt->dt * opt->dt * opt->theta / 2 / opt->dx * it.value());
            }
        for (int k = 0; k < Ms[5].outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Ms[5], k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 2 * opt->Nx, it.col() + 1 * opt->Nx, opt->qom * opt->dt * opt->dt * opt->theta / 2 / opt->dx * it.value());
            }
        for (int k = 0; k < AmpereZ.outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(AmpereZ, k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 2 * opt->Nx, it.col() + 2 * opt->Nx, it.value());
            }
        for (int k = 0; k < Derv.outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Derv, k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 2 * opt->Nx, it.col() + 3 * opt->Nx, -it.value());
            }

        // Fourth Rowblock
        for (int k = 0; k < Derc.outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Derc, k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 3 * opt->Nx, it.col() + 2 * opt->Nx, -it.value());
            }
        for (int k = 0; k < opt->Nx; ++k) {
            tripletListMaxwell[index++] = Triplet<double>(k + 3 * opt->Nx, k + 3 * opt->Nx, 1);
        }

        // Fifth Row
        for (int k = 0; k < Derc.outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(Derc, k); it; ++it)
            {
                tripletListMaxwell[index++] = Triplet<double>(it.row() + 4 * opt->Nx, it.col() + 1 * opt->Nx, it.value());
            }
        for (int k = 0; k < opt->Nx; ++k) {
            tripletListMaxwell[index++] = Triplet<double>(k + 4 * opt->Nx, k + 4 * opt->Nx, 1);
        }
        Maxwell.setFromTriplets(tripletListMaxwell.begin(), tripletListMaxwell.end());
        solver.compute(Maxwell);
        xKrylov = solver.solve(bKrylov);


        E0 = ((Map < Array<double, 3, Dynamic, Eigen::RowMajor>>(xKrylov.data(), 3, opt->Nx)) - E0 * (1 - opt->theta)) / opt->theta;
        Bc.row(1) = (xKrylov(seqN(3 * opt->Nx, opt->Nx)).array() - Bc.row(1).transpose() * (1 - opt->theta)) / opt->theta;
        Bc.row(2) = (xKrylov(seqN(4 * opt->Nx, opt->Nx)).array() - Bc.row(2).transpose() * (1 - opt->theta)) / opt->theta;

        xp += vp.row(0) * opt->dt;

        for (int ip = 0; ip < opt->Np; ip++) {
            vp.col(ip) = (2 * alphap[ip] - Matrix3d::Identity()) * vp.col(ip).matrix() + alphap[ip] * opt->qom * opt->dt * E(t+opt->theta*opt->dt).matrix();
        }
        auto toc = std::chrono::high_resolution_clock::now();
        t += opt->dt;
        //PRINT("Finished timestep", step, " in ", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms using ",opt->Nsub, "subcycles and with relative energy conservation up to = ", diff);
    }
    yn << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    PRINT("t = ", t);
}

int ECSIM_convergence_test(int argc, char* argv[]) {
    int Np = 1; // number of particles
    double L = 2 * EIGEN_PI; // Size of position space

    auto analytical_x = [L](double t) { ArrayXd res(1); res << sin(t); return res; };
    auto analytical_v = [](double t) { Array3d res(3); res << cos(t), -sin(t), cos(t) - sin(t); return res; };
    double k = 3;
    double omega = k;
    auto analytical_E = [L, omega, k](double x, double t) {Array3d res(3); res<< 0, cos(omega * t) * sin(k * x), 0; return res; };
    auto analytical_B = [L, omega, k](double x, double t) {Array3d res(3); res<< 0, 0, -sin(omega * t) * cos(k * x);  return res; };
    int Nx_base = 10; // number of grid cells
    int Nt_base = 10;
    int refinements = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-r" && i + 1 < argc) {
            refinements = std::stoi(argv[++i]);
        }
        else {
            return 1;
        }
    }

    int Nx = 5000;
    ArrayXd xp(Np), qp(Np), xp_end(Np);
    qp(0) = -1;
    Array3Xd vp(3, Np),vp_end(3,Np), E0(3, Nx), Bc(3, Nx), E_analyt(3, Nx), B_analyt(3, Nx);
    std::cout << std::setprecision(16);
    int dimension = (4 * Np + 6 * Nx);
    double t0 = 0;
    double t_end = 1;
    Options opt;
    double dx = L / Nx;
    opt.dx = dx;
    opt.Np = Np;
    opt.Nx = Nx;
    opt.Nsub = 1;
    opt.qom = -1;
    opt.theta = 0.5;
    opt.L = L;
    ArrayXXd errors = ArrayXXd::Zero(refinements, 5);
    ArrayXXd convergence = ArrayXXd::Zero(refinements, 5);


    for (int j = 0; j < Nx; j++) {
        double x = j * dx;
        E0.col(j) = analytical_E(x, t0);
        Bc.col(j) = analytical_B(x + 0.5 * dx, t0);
        E_analyt.col(j) = analytical_E(x, t_end);
        B_analyt.col(j) = analytical_B(x + 0.5 * dx, t_end);
    }
    for (int i = 0; i < refinements; i++) {
        int Nt = Nt_base * pow(2, i);
        double dt = (t_end-t0) / Nt;
        opt.dt = dt;
        opt.Nt = Nt;
        PRINT("TIME CFL = ", dt / L / Nx);

        xp = analytical_x(t0-dt/2);
        vp.col(0) = analytical_v(t0);
        xp_end = analytical_x(t_end - dt / 2);
        vp_end.col(0) = analytical_v(t_end);
        ArrayXd Yn(dimension);
        DecoupledSolve(xp,vp,E0,Bc, t0, t_end, Yn, &opt);
        ArrayXd x_sol = Yn.head(Np);
        Eigen::Map <Array<double, 3, -1>> v_sol(Yn.data() + 1 * Np, 3, Np);
        Eigen::Map <Array<double, 3, -1>> E_sol(Yn.data() + (1 + 3) * Np, 3, Nx);
        Eigen::Map <Array<double, 3, -1>> B_sol(Yn.data() + 1 * Np + (Np + Nx) * 3, 3, Nx);
        errors.row(i) << dt,(analytical_x(t_end - dt / 2) - x_sol).matrix().norm() / (analytical_x(t_end - dt / 2)).matrix().norm(),
            (analytical_v(t_end) - v_sol).matrix().norm() / (analytical_v(t_end)).matrix().norm(),
            (E_analyt - E_sol).matrix().norm() / (E_analyt).matrix().norm(),
            (B_analyt - B_sol).matrix().norm() / B_analyt.matrix().norm();
        if (i > 0)
            convergence.row(i) << dt, errors(i, 1) / errors(i - 1, 1), errors(i, 2) / errors(i - 1, 2), errors(i, 3) / errors(i - 1, 3), errors(i, 4) / errors(i - 1, 4);
    }


    PRINT("TIME ERRORS = ", errors);
    PRINT("TIME CONVERGENCE = ", convergence);
    save("ecsim_time_convergence_errors.txt", errors);

    int Nt = 1000;
    double dt = (t_end - t0) / Nt;
    opt.dt = dt;
    opt.Nt = Nt;

    for (int i = 0; i < refinements; i++) {
        Nx = Nx_base * pow(2, i);
        dx = L / Nx;
        opt.dx = dx;
        opt.Nx = Nx;
        Array3Xd E0(3, Nx), Bc(3, Nx), E_analyt(3, Nx), B_analyt(3, Nx);
        PRINT("SPATIAL CFL = ", dt / dx);
        for (int j = 0; j < Nx; j++) {
            double x = j * dx;
            E0.col(j) = analytical_E(x, t0);
            Bc.col(j) = analytical_B(x + 0.5 * dx, t0);
            E_analyt.col(j) = analytical_E(x, t_end);
            B_analyt.col(j) = analytical_B(x + 0.5 * dx, t_end);
        }
        dimension = (4 * Np + 6 * Nx);
        xp = analytical_x(t0 - dt / 2);
        vp.col(0) = analytical_v(t0);
        xp_end = analytical_x(t_end - dt / 2);
        vp_end.col(0) = analytical_v(t_end);
        ArrayXd Yn(dimension);
        DecoupledSolve(xp, vp, E0, Bc, t0, t_end, Yn, &opt);
        ArrayXd x_sol = Yn.head(Np);
        Eigen::Map <Array<double, 3, -1>> v_sol(Yn.data() + 1 * Np, 3, Np);
        Eigen::Map <Array<double, 3, -1>> E_sol(Yn.data() + (1 + 3) * Np, 3, Nx);
        Eigen::Map <Array<double, 3, -1>> B_sol(Yn.data() + 1 * Np + (Np + Nx) * 3, 3, Nx);
        errors.row(i) << dx, (analytical_x(t_end - dt / 2) - x_sol).matrix().norm() / (analytical_x(t_end - dt / 2)).matrix().norm(),
            (analytical_v(t_end) - v_sol).matrix().norm() / (analytical_v(t_end)).matrix().norm(),
            (E_analyt - E_sol).matrix().norm() / (E_analyt).matrix().norm(),
            (B_analyt - B_sol).matrix().norm() / B_analyt.matrix().norm();
        if (i > 0)
            convergence.row(i) << dx, errors(i, 1) / errors(i - 1, 1), errors(i, 2) / errors(i - 1, 2), errors(i, 3) / errors(i - 1, 3), errors(i, 4) / errors(i - 1, 4);
    }
    PRINT("SPATIAL ERRORS = ", errors);
    PRINT("SPATIAL CONVERGENCE = ", convergence);
    save("ecsim_spatial_convergence_errors.txt", errors);

    return 0;
}

#endif