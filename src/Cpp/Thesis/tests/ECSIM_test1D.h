#ifndef ECSIM_TEST_1D_H
#define ECSIM_TEST_1D_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int ECSIM_1D_test(int argc, char* argv[]) {
    int Nx = 512; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    
    ArrayXd xp(Np), vp(Np), E0(Nx), Bc(Nx),qp(Np);

    TestProblems::SetTwoStream(xp,vp,E0,Bc,qp,Nx,Np,L);

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
        else if (arg == "-c" && i + 1 < argc) {
            coarse_dt = std::stod(argv[++i]);
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
        else {
            std::cerr << "Usage: " << argv[0] << " -f <value> -n <value> -t <value> -c <value> -s <value>" << std::endl;
            return 1;
        }
    }
    auto coarse_solver = ECSIM<1, 1>(L, Np, Nx, Nsub, coarse_dt, qp);
    auto fine_solver = ECSIM<1, 1>(L, Np, Nx, 1, fine_dt, qp);
    auto para_solver = Parareal(fine_solver,coarse_solver);
    int NT = T / coarse_dt;

    VectorXd Xn(2 * Np + 2 * Nx);
    Xn << xp, vp, E0, Bc;

    auto Eold = fine_solver.Energy(Xn);
    VectorXd Yn(2 * Np + 2 * Nx);
    auto tic = std::chrono::high_resolution_clock::now();
    fine_solver.Step(Xn, 0, T, Yn);
    auto toc = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Serial energy difference between beginning and end", abs((fine_solver.Energy(Yn) - Eold).sum()) / abs(Eold.sum()));
    PRINT("Serial took", serial_time, "ms");

    PRINT("------PARAREAL------");
    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
    MatrixXd Xn_para(2 * Np + 2 * Nx, NT + 1);
    Xn_para.col(0) = Xn;
    VectorXd Ediff_para(NT);

    tic = std::chrono::high_resolution_clock::now();
    para_solver.Solve(Xn_para, ts);
    toc = std::chrono::high_resolution_clock::now();
    double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Parareal took", para_time,"ms");
    PRINT("Parareal without subcycling takes", para_time/serial_time, "as long as the serial version");
    PRINT("Energy difference perturbed", abs((fine_solver.Energy(Xn_para.col(NT)) - Eold).sum()) / abs(Eold.sum()));
    PRINT("Diff in energy after simulation = ", abs((fine_solver.Energy(Yn) - fine_solver.Energy(Xn_para.col(NT))).sum()));

    return 0;
}

#endif
