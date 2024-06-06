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
    // Set simulation parameters
    int Nx = 512; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    double coarse_dt = 1e-3;
    double fine_dt = 1e-5;
    double T = 0;
    int num_thr = 12;
    double thresh = 1e-8;
    int subcycling = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c" && i + 1 < argc) {
            coarse_dt = std::stod(argv[++i]);
        }
        if (arg == "-f" && i + 1 < argc) {
            fine_dt = std::stod(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        }
        else if (arg == "-nx" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
        }
        else if (arg == "-n" && i + 1 < argc) {
            num_thr = std::stoi(argv[++i]);
        }
        else if (arg == "-tr" && i + 1 < argc) {
            thresh = std::stod(argv[++i]);
        }
        else if (arg == "-s" && i + 1 < argc) {
            subcycling = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -dt <time step size> -t <time frame [0,t]> -nx <number of gridcells>" << std::endl;
            return 1;
        }
    }
    T = T == 0 ? num_thr * coarse_dt : T;

    // Set initial conditions
    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);
    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);
    MatrixXd Xn_para(4 * Np + 6 * Nx, NT + 1);
    Xn_para.col(0) << xp, vp, E0, Bc;

    // Create fine and coarse solver
    auto fine_solver = ECSIM<1,3>(L, Np, Nx, 1, fine_dt, qp);
	auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, subcycling, coarse_dt, qp);

    // Combine them with parareal
	auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, thresh, num_thr);


    // Perform parareal
    int NT = (int)(T / coarse_dt); // number of time steps
    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
    auto tic = std::chrono::high_resolution_clock::now();
	parareal_solver.Solve(Xn_para, ts);
    auto toc = std::chrono::high_resolution_clock::now();
    PRINT("Finished parareal in", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
	return 0;
}

#endif