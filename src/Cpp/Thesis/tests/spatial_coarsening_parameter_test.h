#ifndef SPATIAL_COARSENING_PARAMETER_TEST_H
#define SPATIAL_COARSENING_PARAMETER_TEST_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>


using namespace Eigen;

int spatial_coarsening_parameter_test(int argc, char* argv[]) {
    int Nx = 32; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles

    double coarse_dt = 1e-2;
    double fine_dt = 1e-4;
    int num_thr = 12;
    double T = 12 * coarse_dt;
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
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -n <number of threads> -t <time interval [0,t]> -c <coarse timestep> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);

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
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, 1, coarse_dt, qp);
    auto fine_solver = ECSIM<1, 3>(L, Np, Nx, 1, fine_dt, qp);
    auto para_solver = Parareal(fine_solver, coarse_solver, thresh);
    int NT = T / coarse_dt;
    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    auto Eold = fine_solver.Energy(Xn);

    MatrixXd Xn_serial(4 * Np + 6 * Nx, NT + 1);
    Xn_serial.col(0) << Xn;
    VectorXd Ediff_fine(NT);

    MatrixXd Xn_para(4 * Np + 6 * Nx, NT + 1);
    Xn_para.col(0) << Xn;
    VectorXd Ediff_para(NT);

    int refinements = 10;
    MatrixXd speedup(refinements, 7);
    for (int j = 0; j < refinements; j++) {
        int fine_Nx = Nx*pow(2,(j));
        PRINT("coarse Nx = ", Nx, ", fine Nx = ", fine_Nx);
        fine_solver.Set_Nx(fine_Nx);

        auto tic = std::chrono::high_resolution_clock::now();
        { // Put it in a scope to save memory
            MatrixXd Xn_serial_fine(4 * Np + 6 * fine_Nx, NT + 1);
            fine_solver.Refine(Xn_serial.col(0), coarse_solver, Xn_serial_fine.col(0));
            for (int i = 0; i < NT; i++) {
                fine_solver.Step(Xn_serial_fine.col(i), ts(i), ts(i + 1), Xn_serial_fine.col(i + 1));
                Ediff_fine(i) = abs((fine_solver.Energy(Xn_serial_fine.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
            }
            coarse_solver.Coarsen(Xn_serial_fine, fine_solver, Xn_serial);
        }
        auto toc = std::chrono::high_resolution_clock::now();
        double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("SERIAL TIME = ", fine_time, "ms");


        tic = std::chrono::high_resolution_clock::now();
        int k = para_solver.Solve(Xn_para, ts);
        for (int i = 0; i < NT; i++) {
            Ediff_para(i) = abs((coarse_solver.Energy(Xn_para.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
        }
        toc = std::chrono::high_resolution_clock::now();
        double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("PARAREAL TIME =", para_time, "ms");


        //PRINT("Finished serial in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        //PRINT("Finished parareal in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        speedup(j, 0) = pow(2, (j));
        speedup(j, 1) = fine_time;
        speedup(j, 2) = para_time;
        speedup(j, 3) = k;
        speedup(j, 4) = coarse_solver.Error(Xn_serial, Xn_para).reshaped(4 * Xn_serial.cols(), 1).maxCoeff();
        speedup(j, 5) = Ediff_para.maxCoeff();
        speedup(j, 6) = fine_time / para_time;
        PRINT("serial time / parareal time = ", speedup(j, 6));
        PRINT("Max relative 2-norm difference between parareal and serial =", speedup(j, 4));
        PRINT("Max relative energy difference against initial state for parareal =", speedup(j, 5));
        save("Parareal_speedup_refining.txt", speedup);
    }
    return 0;
}

#endif