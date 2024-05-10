#ifndef ECSIM_PRECOND_SIM_TEST_H
#define ECSIM_PRECOND_SIM_TEST_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int ECSIM_precond_simulation_test(int argc, char* argv[]) {
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    int Nx = 512; // number of grid cells
    double dt = 1e-2;
    int num_thr = 12;
    double T = 0;
    int Nsub = 1;
    double thresh = 1e-8;
    int refinements = 8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-dt" && i + 1 < argc) {
            dt = std::stod(argv[++i]);
        }
        else if (arg == "-nx" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
        }
        else if (arg == "-s" && i + 1 < argc) {
            Nsub = std::stoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        }
        else if (arg == "-tr" && i + 1 < argc) {
            thresh = std::stod(argv[++i]);
        }
        else if (arg == "-r" && i + 1 < argc) {
            refinements = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -n <number of threads> -t <time interval [0,t]> -c <coarse timestep> -s <number of subcycles> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    T = T == 0 ? num_thr * dt : T;
    PRINT("Simulating time interval = [0, ", T, "], timestep =", dt, "and", Nsub, "subcycles.");

    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);
    int NT = round(T / dt);
    auto id_solver = ECSIM<1, 3, IdentityPreconditioner>(L, Np, Nx, Nsub, dt, qp, LinSolvers::SolverType::GMRES);
    auto dia_solver = ECSIM<1, 3, DiagonalPreconditioner<double>>(L, Np, Nx, Nsub, dt, qp, LinSolvers::SolverType::GMRES);
    auto LU_solver = ECSIM<1, 3, IncompleteLUT<double>>(L, Np, Nx, Nsub, dt, qp, LinSolvers::SolverType::GMRES);

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn.col(0) << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());

    auto Eold = id_solver.Energy(Xn);
    VectorXd Yn(4 * Np + 6 * Nx);
    auto tic = std::chrono::high_resolution_clock::now();
    auto toc = std::chrono::high_resolution_clock::now();

    PRINT("===============IDENTITY===============");
    tic = std::chrono::high_resolution_clock::now();
    auto id_iter = id_solver.Step(Xn, 0, T, Yn);
    toc = std::chrono::high_resolution_clock::now();
    double id_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Identity Preconditioner simulation takes", id_time, "ms and GMRES takes", id_iter.col(2).mean(), "iterations and energy conservation = ", abs((id_solver.Energy(Yn) - Eold).sum()) / abs(Eold.sum()));
    PRINT("time\t timing for solver \t iterations ");
    PRINT(id_iter);

    PRINT("===============JACOBI===============");
    tic = std::chrono::high_resolution_clock::now();
    auto dia_iter = dia_solver.Step(Xn, 0, T, Yn);
    toc = std::chrono::high_resolution_clock::now();
    double dia_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Diagonal Preconditioner simulation takes", dia_time, "ms and GMRES takes", dia_iter.col(2).mean(), "iterations and energy conservation = ", abs((dia_solver.Energy(Yn) - Eold).sum()) / abs(Eold.sum()));
    PRINT("time\t timing for solver \t iterations ");
    PRINT(dia_iter);


    PRINT("===============LU===============");
    tic = std::chrono::high_resolution_clock::now();
    auto lu_iter = LU_solver.Step(Xn, 0, T, Yn);
    toc = std::chrono::high_resolution_clock::now();
    double lu_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Incomplete LU Preconditioner simulation takes", lu_time, "ms and GMRES takes", lu_iter.col(2).mean(), "iterations and energy conservation = ", abs((LU_solver.Energy(Yn) - Eold).sum()) / abs(Eold.sum()));
    PRINT("time\t timing for solver \t iterations ");
    PRINT(lu_iter);

    return 0;
}

#endif
