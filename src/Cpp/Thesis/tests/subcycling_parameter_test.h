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
    int NT = round(T / coarse_dt);
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, 1, coarse_dt, qp, LinSolvers::SolverType::LU);
    auto fine_solver = ECSIM<1, 3>(L, Np, Nx, 1, fine_dt, qp, LinSolvers::SolverType::GMRES);
    auto para_solver = Parareal(fine_solver, coarse_solver, thresh, NT + 1, num_thr);
    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    auto Eold = fine_solver.Energy(Xn);

    // Actually perform the simulations
    // SERIAL
    MatrixXd Xn_fine(4 * Np + 6 * Nx, NT + 1);
    Xn_fine.col(0) << Xn;
    VectorXd Ediff_fine(NT);

    auto tic = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NT; i++) {
        fine_solver.Step(Xn_fine.col(i), ts(i), ts(i + 1), Xn_fine.col(i + 1));
        Ediff_fine(i) = abs((fine_solver.Energy(Xn_fine.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
    }
    auto toc = std::chrono::high_resolution_clock::now();
    double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("SERIAL TIME = ", fine_time, "ms");

    MatrixXd Xn_para(4 * Np + 6 * Nx, NT + 1);
    Xn_para.col(0) <<  Xn;
    VectorXd Ediff_para(NT);

    int refinements = 10;
    MatrixXd speedup(refinements, 11);
    // Get a reference solution
    MatrixXd Yn_ref(Xn.rows(),1);
    double ref_fine_dt = fine_dt / 50;
    fine_solver.Set_dt(ref_fine_dt);
    fine_solver.Step(Xn_para.col(0), 0, T, Yn_ref.col(0));
    fine_solver.Set_dt(fine_dt);
    for (int j = 0; j < refinements; j++) {
        int nsub = j + 1;
        PRINT("subcycles ", nsub);
        coarse_solver.Set_Nsub(nsub);

        tic = std::chrono::high_resolution_clock::now();
        int k = para_solver.Solve(Xn_para, ts);
        toc = std::chrono::high_resolution_clock::now();
        double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("PARAREAL TIME =", para_time, "ms");
        for (int i = 0; i < NT; i++) {
            Ediff_para(i) = abs((fine_solver.Energy(Xn_para.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
        }
        //PRINT("Finished serial in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        //PRINT("Finished parareal in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        speedup(j, 0) = nsub;
        speedup(j, 1) = fine_time;
        speedup(j, 2) = para_time;
        speedup(j, 3) = k;
        speedup(j, 4) = fine_solver.Error(Xn_fine, Xn_para).reshaped(4 * Xn_fine.cols(), 1).maxCoeff();
        speedup(j, 5) = Ediff_para.maxCoeff();
        speedup(j, 6) = fine_time / para_time;
        auto err = fine_solver.Error(Yn_ref, Xn_para.col(NT));
        speedup(j - 1, 7) = err(0);
        speedup(j - 1, 8) = err(1);
        speedup(j - 1, 9) = err(2);
        speedup(j - 1, 10) = err(3);
        PRINT("Speedup = ", speedup(j, 6));
        PRINT("Max relative 2-norm difference between parareal and serial =", speedup(j, 4));
        PRINT("Relative 2-norm difference between parareal and reference =", fine_solver.Error(Yn_ref, Xn_para.col(NT)));
        PRINT("Max relative energy difference against initial state for parareal =", speedup(j, 5));
        save("Parareal_speedup_subcycling.txt", speedup);
    }
    return 0;
}

#endif
