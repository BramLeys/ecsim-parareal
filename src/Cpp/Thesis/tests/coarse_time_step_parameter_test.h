#ifndef COARSE_STEP_PARAMETER_TEST_H
#define COARSE_STEP_PARAMETER_TEST_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>


using namespace Eigen;

int coarse_time_step_parameter_test(int argc, char* argv[]) {
    int Nx = 512; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    double dx = L / Nx; // length of each grid cell
    int Nsub = 1;

    double fine_dt = 1e-4;
    double coarse_dt = 1e-2;
    double thresh = 1e-8;
    int num_thr = 12;
    int refinements = 6;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            fine_dt = std::stod(argv[++i]);
        }
        else if (arg == "-nx" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
        }
        else if (arg == "-n" && i + 1 < argc) {
            num_thr = std::stoi(argv[++i]);
        }
        else if (arg == "-r" && i + 1 < argc) {
            refinements = std::stoi(argv[++i]);
        }
        else if (arg == "-tr" && i + 1 < argc) {
            thresh = std::stod(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -n <number of threads> -nx <number of gridcells> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }

    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);

    auto fine_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, fine_dt, qp, LinSolvers::SolverType::LU);
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, fine_dt, qp, LinSolvers::SolverType::LU);
    auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, thresh, 100, num_thr);

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    fine_solver.Step(Xn, 0, coarse_dt, Xn);


    MatrixXd info(refinements, 8);
    for (int j = 0; j < refinements; j++) {
        int refinement_number = pow(2,j);
        double coarse_dt = fine_dt * refinement_number;
        double T = num_thr*coarse_dt;
        PRINT("dt_fine = ", fine_dt, ", dt_coarse = ", coarse_dt);
        int NT = round(T / coarse_dt); 

        MatrixXd Xn_fine(Xn.rows(), NT + 1);
        Xn_fine.col(0) << Xn;
        auto tic = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NT; i++) {
            fine_solver.Step(Xn_fine.col(i), i * coarse_dt, (i + 1) * coarse_dt, Xn_fine.col(i + 1));
        }
        auto toc = std::chrono::high_resolution_clock::now();
        double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("SERIAL TIME =", fine_time, "ms");

        VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
        MatrixXd Xn_para(Xn.rows(), NT + 1);
        Xn_para.col(0) << Xn;

        coarse_solver.Set_dt(coarse_dt);

        tic = std::chrono::high_resolution_clock::now();
        int k = parareal_solver.Solve(Xn_para, ts);
        toc = std::chrono::high_resolution_clock::now();
        double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("PARAREAL TIME =", para_time, "ms");
        info(j, 0) = refinement_number;
        info(j, 1) = fine_time;
        info(j, 2) = para_time;
        info(j, 3) = k;
        info(j, 4) = fine_solver.Error(Xn_fine, Xn_para).reshaped(4 * Xn_para.cols(), 1).maxCoeff();
        info(j, 5) = fine_time / para_time;
        info(j, 6) = coarse_dt;
        info(j, 7) = Nx;
        PRINT("SPEEDUP = ", info(j, 5));
        PRINT("Max relative 2-norm difference between parareal and serial =", info(j, 4));
        save("coarse_time_step_test.txt", info);
    }
    return 0;
}

#endif