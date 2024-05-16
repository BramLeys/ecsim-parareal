#ifndef SUBCYCLING_PARAMETER_TEST_H
#define SUBCYCLING_PARAMETER_TEST_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>
#include <cstdlib>


using namespace Eigen;

int subcycling_parameter_test(int argc, char* argv[]) {

    int Nx = 512; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles

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

    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);
    int NT = round(T / coarse_dt);
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, 1, coarse_dt, qp, LinSolvers::SolverType::GMRES);
    auto fine_solver = ECSIM<1, 3>(L, Np, Nx, 1, fine_dt, qp, LinSolvers::SolverType::GMRES);
    auto para_solver = Parareal(fine_solver, coarse_solver, thresh, NT + 1, num_thr);
    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());

    // SERIAL
    MatrixXd Xn_fine(4 * Np + 6 * Nx, NT + 1);
    Xn_fine.col(0) << Xn;

    auto tic = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NT; i++) {
        fine_solver.Step(Xn_fine.col(i), ts(i), ts(i + 1), Xn_fine.col(i + 1));
    }
    auto toc = std::chrono::high_resolution_clock::now();
    double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("SERIAL TIME = ", fine_time, "ms");

    MatrixXd Xn_para(4 * Np + 6 * Nx, NT + 1);
    Xn_para.col(0) <<  Xn;

    int refinements = 10;
    MatrixXd info(refinements, 9);
    for (int j = 0; j < refinements; j++) {
        int nsub = j + 1;
        PRINT("subcycles ", nsub);
        coarse_solver.Set_Nsub(nsub);
        tic = std::chrono::high_resolution_clock::now();
        int k = para_solver.Solve(Xn_para, ts);
        toc = std::chrono::high_resolution_clock::now();
        double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("PARAREAL TIME =", para_time, "ms");
        info(j, 0) = nsub;
        info(j, 1) = fine_time;
        info(j, 2) = para_time;
        info(j, 3) = k;
        info(j, 4) = fine_solver.Error(Xn_fine, Xn_para).maxCoeff();
        info(j, 5) = fine_time / para_time;
        info(j, 6) = coarse_dt;
        info(j, 7) = fine_dt;
        info(j, 8) = Nx;
        PRINT("Speedup = ", info(j, 5));
        PRINT("Max relative 2-norm difference between parareal and serial =", info(j, 4));
        save("subcycling_test.txt", info);
    }
    return 0;
}

#endif
