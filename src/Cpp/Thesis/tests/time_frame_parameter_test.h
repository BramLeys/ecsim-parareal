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
    int num_thr = 12;
    double T = 0;
    double thresh = 1e-8;

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
        else if (arg == "-n" && i + 1 < argc) {
            num_thr = std::stoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        }
        else if (arg == "-tr" && i + 1 < argc) {
            thresh = std::stod(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -n <number of threads> -nx <number of gridpoints> -t <time interval [0,t]> -c <coarse timestep> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    T = T == 0 ? num_thr * coarse_dt : T;

    ArrayXd xp(Np), vp(Np), E0(Nx), Bc(Nx), qp(Np);

    TestProblems::SetTwoStream(xp, vp, E0, Bc, qp, Nx, Np, L);

    auto fine_solver = ECSIM<1, 1>(L, Np, Nx, Nsub, fine_dt, qp, LinSolvers::SolverType::GMRES);
    auto coarse_solver = ECSIM<1, 1>(L, Np, Nx, Nsub, coarse_dt, qp, LinSolvers::SolverType::LU);
    auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, thresh, 50, num_thr);
    int refinements = 8;
    MatrixXd speedup(refinements, 7);
    for (int j = 1; j <= refinements; j++) {
        int refinement_number = 12 * j;
        double T = refinement_number * coarse_dt;
        PRINT("Testing time frame =",T);
        int NT = round(T / coarse_dt); // number of time steps
        VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
        MatrixXd Xn_fine(2 * Np + 2 * Nx, NT + 1);
        Xn_fine.col(0) << xp, vp, E0, Bc;
        auto Eold = fine_solver.Energy(Xn_fine.col(0));

        VectorXd Ediff_fine(NT);
        MatrixXd Xn_para(2 * Np + 2 * Nx, NT + 1);
        Xn_para.col(0) << xp, vp, E0, Bc;

        auto tic = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NT; i++) {
            fine_solver.Step(Xn_fine.col(i), ts(i), ts(i + 1), Xn_fine.col(i + 1));
            Ediff_fine(i) = abs((fine_solver.Energy(Xn_fine.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
        }
        auto toc = std::chrono::high_resolution_clock::now();
        double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();

        tic = std::chrono::high_resolution_clock::now();
        int k = parareal_solver.Solve(Xn_para, ts);
        toc = std::chrono::high_resolution_clock::now();
        double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        VectorXd Ediff_para(NT);
        for (int i = 0; i < NT; i++) {
            Ediff_para(i) = abs((fine_solver.Energy(Xn_para.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
        }
        //PRINT("Finished serial in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        //PRINT("Finished parareal in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        speedup(j - 1, 0) = refinement_number;
        speedup(j - 1, 1) = fine_time;
        speedup(j - 1, 2) = para_time;
        speedup(j - 1, 3) = k;
        speedup(j - 1, 4) = fine_solver.Error(Xn_fine, Xn_para).reshaped(4 * Xn_fine.cols(), 1).maxCoeff();
        speedup(j - 1, 5) = Ediff_para.maxCoeff();
        speedup(j - 1, 6) = fine_time / para_time;
        PRINT("Speedup = ", speedup(j - 1, 6));
        PRINT("Max relative 2-norm difference between parareal and serial =", speedup(j - 1, 4));
        PRINT("Max relative energy difference against initial state for parareal =", speedup(j - 1, 5));
        save("Parareal_speedup_time_interval.mat", speedup);
    }
    return 0;
}

#endif