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
    // Randomize
    srand((unsigned int)time(0));

    //Initialize the problem (setting initial state)
    int Nx = 128; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    ArrayXd xp(Np), vp(Np), E0(Nx), Bc(Nx), qp(Np);

    TestProblems::SetTwoStream(xp, vp, E0, Bc, qp, Nx, Np, L);

    // Define the solvers
    double coarse_dt = 1e-2;
    double fine_dt = 1e-4;
    int num_thr = 12;
    double T = 12 * coarse_dt;
    int Nsub = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            fine_dt = std::stod(argv[++i]);
        }
        else if (arg == "-n" && i + 1 < argc) {
            num_thr = std::stoi(argv[++i]);
        }
        else if (arg == "-s" && i + 1 < argc) {
            Nsub = std::stoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        }
        else if (arg == "-c" && i + 1 < argc) {
            coarse_dt = std::stod(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <value> -n <value> -t <value> -c <value> -s <value>" << std::endl;
            return 1;
        }
    }

    int nthreads, tid;
    #pragma omp parallel private(nthreads, tid) num_threads(num_thr)
    {
        /* Obtain thread number */
        tid = omp_get_thread_num();
        /* Only master thread does this */
        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }

    }  /* All threads join master thread and disband */

    double wp_P = 1e-8;
    auto fine_solver = ECSIM<1, 1>(L, Np, Nx, 1, fine_dt, qp);
    auto coarse_solver = ECSIM<1, 1>(L, Np, Nx, 1, coarse_dt, qp);
    auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, wp_P, 50, num_thr);
    int NT = (int)(T / coarse_dt); // number of time steps
    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);

    // Actually perform the simulations
    // SERIAL
    MatrixXd Xn_fine(2 * Np + 2 * Nx, NT + 1);
    Xn_fine.col(0) << xp, vp, E0, Bc;
    VectorXd Ediff_fine(NT);
    auto Eold = fine_solver.Energy(Xn_fine.col(0));

    auto tic = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NT; i++) {
        fine_solver.Step(Xn_fine.col(i), ts(i), ts(i + 1), Xn_fine.col(i + 1));
        Ediff_fine(i) = abs((fine_solver.Energy(Xn_fine.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
    }
    auto toc = std::chrono::high_resolution_clock::now();
    double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();

    // PARAREAL without subcycling
    MatrixXd Xn_para(2 * Np + 2 * Nx, NT + 1);
    Xn_para.col(0) << xp, vp, E0, Bc;
    VectorXd Ediff_para(NT);

    tic = std::chrono::high_resolution_clock::now();
    parareal_solver.Solve(Xn_para, ts);
    toc = std::chrono::high_resolution_clock::now();
    double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    for (int i = 0; i < NT; i++) {
        Ediff_para(i) = abs((fine_solver.Energy(Xn_para.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
    }
    PRINT("Parareal without subcycling takes", para_time / fine_time, "as long as the serial version");
    PRINT("Max relative energy difference against initial state for parareal without subcycling =", Ediff_para.maxCoeff());
    PRINT("Relative 2-norm difference between parareal without subcycling and serial =", (Xn_fine - Xn_para).norm() / Xn_fine.norm());

    // PARAREAL with subcycling
    MatrixXd Xn_para_sub(2 * Np + 2 * Nx, NT + 1);
    Xn_para_sub.col(0) << xp, vp, E0, Bc;
    VectorXd Ediff_para_sub(NT);
    coarse_solver.Set_Nsub(Nsub);

    tic = std::chrono::high_resolution_clock::now();
    parareal_solver.Solve(Xn_para_sub, ts);
    toc = std::chrono::high_resolution_clock::now();
    double para_sub_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    for (int i = 0; i < NT; i++) {
        Ediff_para_sub(i) = abs((fine_solver.Energy(Xn_para_sub.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
    }
    PRINT("Parareal with subcycling takes", para_sub_time / fine_time, "as long as the serial version");
    PRINT("Max relative energy difference against initial state for parareal with subcycling =", Ediff_para_sub.maxCoeff());
    PRINT("Relative 2-norm difference between parareal with subcycling and serial =", (Xn_fine - Xn_para_sub).norm() / Xn_fine.norm());

    return 0;
}

#endif
