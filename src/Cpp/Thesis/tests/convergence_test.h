#ifndef CONVERGENCE_TEST_3D_H
#define CONVERGENCE_TEST_3D_H

#include "../parareal.h"
#include "../common.h"
#include "../solvers.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int convergence_test(int argc, char* argv[]) {
    int N = 10;
    double T = 0;
    double dt = 1e-1;
    double thresh = 1e-8;
    double L = 2 * EIGEN_PI;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-dt" && i + 1 < argc) {
            dt = std::stod(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        }
        else if (arg == "-tr" && i + 1 < argc) {
            thresh = std::stod(argv[++i]);
        }
        else if (arg == "-N" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -t <time interval [0,t]> -c <coarse timestep> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    T = T == 0 ? 12 * dt : T;
    double dx = L / N;
    auto first_order = [&dx](const Ref<const MatrixXd> xn, double tn, Ref<MatrixXd> yn) {
        for (int j = 0; j < yn.cols(); j++) {
            for (int i = 0; i < yn.rows(); i++) {
                yn(i, j) = (-xn((i - 1 + yn.rows()) % yn.rows(), j) + xn((i + 1) % yn.rows(), j)) / (2 * dx);
            }
        }
        };
    auto second_order = [&dx](const Ref<const MatrixXd> xn, double tn, Ref<MatrixXd> yn) {
        for (int j = 0; j < yn.cols(); j++) {
            for (int i = 0; i < yn.rows(); i++) {
                yn(i, j) = (xn((i - 1 + yn.rows()) % yn.rows(), j) - 2 * xn(i, j) + xn((i + 1) % yn.rows(), j)) / (dx * dx);
            }
        }
        };

    Eigen::SparseMatrix<double> A(N, N);
    A.reserve(Eigen::VectorXi::Constant(N, 2));
    for (int i = 0; i < N; ++i) {
        A.insert(i, (i + 1) % N) = 1 / (2 * dx);
        A.insert(i, (i - 1 + N) % N) = -1 / (2 * dx);
    }
    A.makeCompressed();

    Eigen::SparseMatrix<double> B(N, N);
    B.reserve(Eigen::VectorXi::Constant(N, 3));
    for (int i = 0; i < N; ++i) {
        B.insert(i, (i + 1) % N) = 1 / ( dx * dx);
        B.insert(i, i) = - 2 / (dx * dx);
        B.insert(i, (i - 1 + N) % N) = 1 / ( dx * dx);
    }
    B.makeCompressed();

    int refinements = 10;
    auto solverRef = CrankNicolson(B, dt/pow(2,refinements+3));
    auto solverCN = CrankNicolson(B, dt);
    auto solverRK4 = RK4<decltype(second_order)>(second_order, dt);

    VectorXd X = VectorXd::LinSpaced(N,0,L);
    VectorXd Yn_ref(N), Yn_CN(N), Yn_RK4(N);
    solverRef.Step(X, 0, T, Yn_ref);

    ArrayXXd errors(refinements, 3);
    for (int i = 0; i < refinements; i++) {
        Yn_CN = VectorXd::Zero(N);
        double refined_dt = dt / pow(2, i);
        solverCN.Set_dt(refined_dt);
        solverRK4.Set_dt(refined_dt);

        solverCN.Step(X, 0, T, Yn_CN);
        solverRK4.Step(X, 0, T, Yn_RK4);
        PRINT("dt = ", refined_dt);
        PRINT("dt/dx^2 = ", refined_dt / dx/dx);
        errors(i, 0) = refined_dt;
        errors(i, 1) = (Yn_CN - Yn_ref).norm() / Yn_ref.norm();
        errors(i, 2) = (Yn_RK4 - Yn_ref).norm() / Yn_ref.norm();
        PRINT("CN error = ", errors(i, 1));
        PRINT("RK4 error = ", errors(i, 2));
    }

    PRINT("Errors:", errors);
    save("convergence_errors.txt", errors);
    return 0;
}

#endif