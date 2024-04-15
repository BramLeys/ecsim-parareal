#ifndef TIME_STEP_PARAMETER_TEST_H
#define TIME_STEP_PARAMETER_TEST_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>


using namespace Eigen;

int time_step_parameter_test(int argc, char* argv[]) {

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
    int NT = round(T / coarse_dt);

    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);
    auto fine_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, coarse_dt, qp, LinSolvers::SolverType::GMRES);
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, coarse_dt, qp, LinSolvers::SolverType::LU);
    auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, thresh, NT + 1, num_thr);

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    fine_solver.Step(Xn, 0, coarse_dt, Xn);
    auto Eold = fine_solver.Energy(Xn);

    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
    MatrixXd Xn_fine(Xn.rows(), NT + 1);
    Xn_fine.col(0) << Xn;
    MatrixXd Xn_para(Xn.rows(), NT + 1);
    Xn_para.col(0) << Xn;

    int refinements = 20;
    VectorXd Ediff_fine(NT);
    VectorXd Ediff_para(NT);
    MatrixXd speedup(refinements,7);
    for (int j = 1; j <= refinements; j++) {
        int refinement_number = 10 * j;
        double fine_dt = coarse_dt / refinement_number;
        PRINT("dt_fine = ", refinement_number, "* dt_coarse");
        fine_solver.Set_dt(fine_dt);

        auto tic = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NT; i++) {
            fine_solver.Step(Xn_fine.col(i), ts(i), ts(i + 1), Xn_fine.col(i + 1));
            Ediff_fine(i) = abs((fine_solver.Energy(Xn_fine.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
        }
        auto toc = std::chrono::high_resolution_clock::now();
        double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("SERIAL TIME =", fine_time, "ms");

        tic = std::chrono::high_resolution_clock::now();
        int k = parareal_solver.Solve(Xn_para, ts);
        toc = std::chrono::high_resolution_clock::now();
        double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("PARAREAL TIME =", para_time, "ms");
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
        save("Parareal_speedup_fine_time_steps.txt", speedup);
    }
    //save("Parareal_v_evolution.txt", Xn_para.block(Np, 0,Np,Xn_para.cols()).transpose());
    return 0;
}

#endif