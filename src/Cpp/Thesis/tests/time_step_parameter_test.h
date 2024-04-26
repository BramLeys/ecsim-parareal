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
    int refinements = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            fine_dt = std::stod(argv[++i]);
        }
        else if (arg == "-c" && i + 1 < argc) {
            coarse_dt = std::stod(argv[++i]);
        }
        else if (arg == "-r" && i + 1 < argc) {
            refinements = std::stoi(argv[++i]);
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
    auto fine_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, coarse_dt, qp, LinSolvers::SolverType::LU);
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

    VectorXd Ediff_fine(NT);
    VectorXd Ediff_para(NT);
    MatrixXd speedup(refinements + 1,7);

    // Get a reference solution
    MatrixXd Yn_ref(Xn.rows(), 1);
    fine_dt = coarse_dt / (pow(2, refinements+3));
    fine_solver.Set_dt(fine_dt);
    fine_solver.Step(Xn_para.col(0), 0, T, Yn_ref.col(0));
    ArrayXXd error(refinements+1, 4);
    ArrayXXd convergence(refinements+1, 4);
    for (int j = 0; j <= refinements; j++) {
        int refinement_number = pow(2,j);
        fine_dt = coarse_dt / refinement_number;
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
        speedup(j, 0) = refinement_number;
        speedup(j, 1) = fine_time;
        speedup(j, 2) = para_time;
        speedup(j, 3) = k;
        speedup(j, 4) = fine_solver.Error(Xn_fine, Xn_para).reshaped(4 * Xn_fine.cols(), 1).maxCoeff();
        speedup(j, 5) = Ediff_para.maxCoeff();
        speedup(j, 6) = fine_time / para_time;
        auto err = fine_solver.Error(Yn_ref, Xn_para.col(NT));
        error.row(j) = err.transpose();
        if (j > 0) {
            convergence.row(j) = error.row(j) / error.row(j - 1);
        }
        PRINT("Speedup = ", speedup(j, 6));
        PRINT("Max relative 2-norm difference between parareal and serial =", speedup(j, 4));
        PRINT("Relative 2-norm difference between parareal and reference =", fine_solver.Error(Yn_ref, Xn_para.col(NT)));
        PRINT("Max relative energy difference against initial state for parareal =", speedup(j, 5));
        save("Parareal_speedup_fine_time_steps.txt", speedup, "\trefinement number \t serial time \t parareal time \t number of parareal iterations \t max parareal error against serial solve \t max parareal energy diff \t speedup \t x error against ref \t v error against ref \t E error against ref \t B error against ref");
    }
    PRINT("ERROR: ", error);
    PRINT("CONVERGENCE: ", convergence);
    save("convergence_parareal.txt", convergence);
    save("Parareal_sim.txt", Xn_para);
    return 0;
}

#endif