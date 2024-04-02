#ifndef ECSIM_PARAREAL_TEST_H
#define ECSIM_PARAREAL_TEST_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int ECSIM_parareal_test(int argc, char* argv[]) {
    int Nx = 128; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    int Nsub = 1;
    ArrayXd xp(Np), vp(Np), E0(Nx), Bc(Nx), qp(Np);

    TestProblems::SetTwoStream(xp, vp, E0, Bc, qp, Nx, Np, L);

    double coarse_dt = atof(argv[1]);
    double fine_dt = atof(argv[2]);
    double wp_P = 1e-8;

    double time_interval = atof(argv[3]);
    int fine_subcycling = 1;
    int coarse_subcycling = 1*fine_subcycling;
    double T = time_interval/coarse_subcycling;

    int NT = (int)(T / coarse_dt); // number of time steps

    auto fine_solver = ECSIM<1,1>(L, Np, Nx, fine_subcycling, fine_dt, qp);
	auto coarse_solver = ECSIM<1, 1>(L, Np, Nx, coarse_subcycling, coarse_dt, qp);
	auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, wp_P, 10);
    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);

    /*VectorXd Ediff_fine(NT);
    MatrixXd Xn_fine(2 * Np + 2*Nx, NT+1);
    Xn_fine.col(0) << xp, vp, E0, Bc;
    auto Eold = fine_solver.Energy(Xn_fine.col(0));

    auto tic = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NT; i++) {
        fine_solver.Step(Xn_fine.col(i), ts(i), ts(i+1), Xn_fine.col(i + 1));
        Ediff_fine(i) = abs((fine_solver.Energy(Xn_fine.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
    }
    auto toc = std::chrono::high_resolution_clock::now();
    PRINT("Finished serial in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
    PRINT("Max relative energy difference against initial state for serial =", Ediff_fine.maxCoeff());*/

    //MatrixXd Xn_fine(2 * Np + 2 * Nx, NT + 1);
    //Xn_fine.col(0) << xp, vp, E0, Bc;
    //auto Eold = fine_solver.Energy(Xn_fine.col(0));
    //auto tic = std::chrono::high_resolution_clock::now();
    //fine_solver.Solve(Xn_fine.col(0), 0, T, Xn_fine);
    //auto toc = std::chrono::high_resolution_clock::now();
    //PRINT("Finished serial in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
    //PRINT("Serial difference in energy between initial and final states =", abs((fine_solver.Energy(Xn_fine.col(NT)) - Eold).sum()) / abs(Eold.sum()));

    MatrixXd Xn_para(2 * Np + 2 * Nx, NT + 1);
    Xn_para.col(0) << xp, vp, E0, Bc;
    //tic = std::chrono::high_resolution_clock::now();
	parareal_solver.Solve(Xn_para, ts);
    //toc = std::chrono::high_resolution_clock::now();
    /*VectorXd Ediff_para(NT);
    for (int i = 0; i < NT; i++) {
        Ediff_para(i) = abs((fine_solver.Energy(Xn_para.col(i + 1)) - Eold).sum()) / abs(Eold.sum());
    }*/
    //PRINT("Finished parareal in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
    //PRINT("Max relative energy difference against initial state for parareal =", Ediff_para.maxCoeff());
    //PRINT("Relative 2-norm difference between parareal and serial =", (Xn_fine - Xn_para).norm() / Xn_fine.norm());
	return 0;
}

#endif