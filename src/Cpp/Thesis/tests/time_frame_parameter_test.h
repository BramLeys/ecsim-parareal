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
    int Nx = 128; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    double dx = L / Nx; // length of each grid cell
    int Nsub = 1;
    ArrayXd xp(Np), vp(Np), E0(Nx), Bc(Nx), qp(Np);

    TestProblems::SetTwoStream(xp, vp, E0, Bc, qp, Nx, Np, L);

    double coarse_dt = 1e-2;
    double fine_dt = coarse_dt / 100;
    double wp_P = 1e-8;
    auto fine_solver = ECSIM<1, 1>(L, Np, Nx, Nsub, fine_dt, qp);
    auto coarse_solver = ECSIM<1, 1>(L, Np, Nx, Nsub, coarse_dt, qp);
    auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, wp_P, 50);
    int refinements = 50;
    MatrixXd speedup(refinements, 2);
    for (int j = 1; j <= refinements; j++) {
        double T = 12*j*coarse_dt;
        PRINT("Testing time frame =",T);
        int NT = (int)(T / coarse_dt); // number of time steps
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
        parareal_solver.Solve(Xn_para, ts);
        toc = std::chrono::high_resolution_clock::now();
        double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        VectorXd Ediff_para(NT);
        for (int i = 0; i < NT; i++) {
            Ediff_para(i) = abs((fine_solver.Energy(Xn_para.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
        }
        //PRINT("Finished serial in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        //PRINT("Finished parareal in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
        PRINT("Parareal takes", para_time / fine_time, "as long as the serial version");
        speedup(j - 1, 0) = 12 * j;
        speedup(j - 1, 1) = para_time / fine_time;
        PRINT("Max relative energy difference against initial state for parareal =", Ediff_para.maxCoeff());
        PRINT("Relative 2-norm difference between parareal and serial =", (Xn_fine - Xn_para).norm() / Xn_fine.norm());
        save("Parareal_speedup_time_interval.mat", speedup);
    }
    return 0;
}

#endif