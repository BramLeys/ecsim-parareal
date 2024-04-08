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
    int Nx = 128; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    double dx = L / Nx; // length of each grid cell
    int Nsub = 1;

    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);

    double fine_dt = 1e-4;
    double coarse_dt = 1e-2;
    double T = 1;
    double wp_P = 1e-8;
    auto fine_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, fine_dt, qp, LinSolvers::SolverType::GMRES);
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, fine_dt, qp, LinSolvers::SolverType::LU);
    auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, wp_P, 50);

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    fine_solver.Step(Xn, 0, 100*fine_dt, Xn);
    auto Eold = fine_solver.Energy(Xn);

    int NT_fine = int(T / fine_dt);

    MatrixXd Xn_fine(Xn.rows(), NT_fine + 1);
    Xn_fine.col(0) << Xn;
    VectorXd Ediff_fine(NT_fine);
    auto tic = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NT_fine; i++) {
        fine_solver.Step(Xn_fine.col(i), i * fine_dt, (i + 1) * fine_dt, Xn_fine.col(i + 1));
        Ediff_fine(i) = abs((fine_solver.Energy(Xn_fine.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
    }
    auto toc = std::chrono::high_resolution_clock::now();
    double fine_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("SERIAL TIME =", fine_time, "ms");

    int refinements = 10;
    MatrixXd speedup(refinements, 6);
    for (int j = 1; j <= refinements; j++) {
        int refinement_number = j * 100;
        double coarse_dt = fine_dt * refinement_number;
        PRINT("dt_coarse = ", refinement_number, "* dt_coarse");
        int NT = int(T / coarse_dt); // number of time steps


        VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
        MatrixXd Xn_para(Xn.rows(), NT + 1);
        MatrixXd fine_view = MatrixXd(Xn.rows(), NT + 1);
        Xn_para.col(0) << Xn;
        fine_view.col(0) << Xn;
        VectorXd Ediff_para(NT);

        coarse_solver.Set_dt(coarse_dt);

        tic = std::chrono::high_resolution_clock::now();
        int k = parareal_solver.Solve(Xn_para, ts);
        toc = std::chrono::high_resolution_clock::now();
        double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("PARAREAL TIME =", para_time, "ms");
        for (int i = 0; i < NT; i++) {
            int fine_step = (i + 1) * refinement_number;
            fine_view.col(i + 1) = Xn_fine.col(fine_step);
            Ediff_para(i) = abs((fine_solver.Energy(Xn_para.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
        }
        //PRINT("Finished serial in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        //PRINT("Finished parareal in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        PRINT("Parareal takes", para_time / fine_time, "as long as the serial version");
        speedup(j - 1, 0) = refinement_number;
        speedup(j - 1, 1) = fine_time;
        speedup(j - 1, 2) = para_time;
        speedup(j - 1, 3) = k;
        speedup(j - 1, 4) = fine_solver.Error(fine_view, Xn_para).reshaped(4 * Xn_para.cols(), 1).maxCoeff();
        speedup(j - 1, 5) = Ediff_para.maxCoeff();
        PRINT("Max relative 2-norm difference between parareal and serial =", speedup(j - 1, 4));
        PRINT("Max relative energy difference against initial state for parareal =", speedup(j - 1, 5));
        save("Parareal_speedup_coarse_time_steps.txt", speedup);
    }
    //save("Parareal_v_evolution.txt", Xn_para.block(Np, 0,Np,Xn_para.cols()).transpose());
    return 0;
}

#endif