#ifndef TIME_FRAME_PARAMETER_TEST_H
#define TIME_FRAME_PARAMETER_TEST_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>


using namespace Eigen;

int time_frame_parameter_test(int argc, char* argv[]) {
    int Nx = 512; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    double dx = L / Nx; // length of each grid cell
    int Nsub = 1;

    double coarse_dt = 1e-3;
    double fine_dt = 1e-5;
    double thresh = 1e-8;
    int refinements = 8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            fine_dt = std::stod(argv[++i]);
        }
        else if (arg == "-c" && i + 1 < argc) {
            coarse_dt = std::stod(argv[++i]);
        }
        else if (arg == "-nx" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
        }

        else if (arg == "-r" && i + 1 < argc) {
            refinements = std::stoi(argv[++i]);
        }
        else if (arg == "-tr" && i + 1 < argc) {
            thresh = std::stod(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -n <number of threads> -nx <number of gridpoints> -r <number of refinements> -c <coarse timestep> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }

    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);

    int dim = 4 * Np + 6 * Nx;

    auto fine_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, fine_dt, qp, LinSolvers::SolverType::GMRES);
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, coarse_dt, qp, LinSolvers::SolverType::GMRES);
    auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, thresh, 100, 1);
    MatrixXd info(refinements, 9);
    for (int j = 1; j <= refinements; j++) {
        int refinement_number = 6 * j;
        double T = refinement_number * coarse_dt;
        PRINT("Testing time frame =",T);
        parareal_solver.setNumThreads(refinement_number);
        int NT = round(T / coarse_dt); // number of time steps
        VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
        MatrixXd Xn_fine(dim, NT + 1);
        Xn_fine.col(0) << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());

        MatrixXd Xn_para(dim, NT + 1);
        Xn_para.col(0) << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());

        auto tic = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NT; i++) {
            fine_solver.Step(Xn_fine.col(i), ts(i), ts(i + 1), Xn_fine.col(i + 1));
        }
        auto toc = std::chrono::high_resolution_clock::now();
        double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();

        tic = std::chrono::high_resolution_clock::now();
        int k = parareal_solver.Solve(Xn_para, ts);
        toc = std::chrono::high_resolution_clock::now();
        double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        info(j - 1, 0) = refinement_number;
        info(j - 1, 1) = fine_time;
        info(j - 1, 2) = para_time;
        info(j - 1, 3) = k;
        info(j - 1, 4) = fine_solver.Error(Xn_fine, Xn_para).maxCoeff();
        info(j - 1, 5) = fine_time / para_time;
        info(j - 1, 6) = coarse_dt;
        info(j - 1, 7) = fine_dt;
        info(j - 1, 8) = Nx;
        PRINT("Speedup = ", info(j - 1, 5));
        PRINT("Max relative 2-norm difference between parareal and serial =", info(j - 1, 4));
        save("core_scaling_test.txt", info);
    }
    return 0;
}

#endif