#ifndef ECSIM_CHAOTIC_TEST_1D_H
#define ECSIM_CHAOTIC_TEST_1D_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int ECSIM_1D_chaotic_test(int argc, char* argv[]) {
    int Nx = 128; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 100000; // number of particles
    int Nsub = 1;

    ArrayXd xp(Np), vp(Np), E0(Nx), Bc(Nx), qp(Np);

    TestProblems::SetTwoStream(xp, vp, E0, Bc, qp, Nx, Np, L);

    double T = std::stod(argv[1]);
    double coarse_dt = std::stod(argv[2]);
    double fine_dt = std::stod(argv[3]);
    auto coarse_solver = ECSIM<1, 1>(L, Np, Nx, 1, coarse_dt, qp, LinSolvers::SolverType::LU);
    auto fine_solver = ECSIM<1, 1>(L, Np, Nx, 1, fine_dt, qp, LinSolvers::SolverType::GMRES);
    auto para_solver = Parareal(fine_solver, coarse_solver);
    int NT = T / coarse_dt;

    VectorXd Xn(2 * Np + 2 * Nx);
    Xn << xp, vp, E0, Bc;

    auto Eold = fine_solver.Energy(Xn);
    VectorXd Yn(2 * Np + 2 * Nx);
    fine_solver.Step(Xn, 0, T, Yn);
    PRINT("Energy difference", abs((fine_solver.Energy(Yn) - Eold).sum()) / abs(Eold.sum()));

    VectorXd perturb = VectorXd::Constant(Xn.rows(), Xn.cols(), 1e-8);
    VectorXd Xn2 = Xn + perturb;
    VectorXd Yn2(2 * Np + 2 * Nx);
    auto Eold2 = fine_solver.Energy(Xn2);
    PRINT("Initial Energy diff =", abs(Eold2.sum() - Eold.sum()), "Initial diff in states = ", fine_solver.Error(Xn, Xn2));
    auto tic = std::chrono::high_resolution_clock::now();
    fine_solver.Step(Xn2, 0, T, Yn2);
    auto toc = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("------SERIAL------");
    PRINT("Energy difference perturbed", abs((fine_solver.Energy(Yn2) - Eold2).sum()) / abs(Eold2.sum()));
    PRINT("Diff in energy after simulation = ", abs((fine_solver.Energy(Yn) - fine_solver.Energy(Yn2)).sum()), "Diff in states after simulation =", fine_solver.Error(Yn, Yn2));

    PRINT("------PARAREAL------");
    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
    MatrixXd Xn_para(2 * Np + 2 * Nx, NT + 1);
    Xn_para.col(0) = Xn2;
    VectorXd Ediff_para(NT);

    tic = std::chrono::high_resolution_clock::now();
    para_solver.Solve(Xn_para, ts);
    toc = std::chrono::high_resolution_clock::now();
    double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Parareal without subcycling takes", para_time / serial_time, "as long as the serial version");
    PRINT("Energy difference perturbed", abs((fine_solver.Energy(Xn_para.col(NT)) - Eold2).sum()) / abs(Eold2.sum()));
    PRINT("Diff in energy after simulation = ", abs((fine_solver.Energy(Yn) - fine_solver.Energy(Xn_para.col(NT))).sum()), "Diff in states after simulation for perturbed versions =", fine_solver.Error(Xn_para.col(NT), Yn2));

    return 0;
}

#endif
