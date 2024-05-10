#ifndef ECSIM_TEST_3D_H
#define ECSIM_TEST_3D_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int ECSIM_3D_test(int argc, char* argv[]) {
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    int Nx = 512; // number of grid cells
    int fine_Nx = Nx;
    double coarse_dt = 1e-2;
    double fine_dt = 1e-4;
    int num_thr = 12;
    double T = 0;
    int Nsub = 1;
    double thresh = 1e-8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            fine_dt = std::stod(argv[++i]);
        }
        else if (arg == "-c" && i + 1 < argc) {
            coarse_dt = std::stod(argv[++i]);
        }
        else if (arg == "-nxc" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
        }
        else if (arg == "-nxf" && i + 1 < argc) {
            fine_Nx = std::stoi(argv[++i]);
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
        else if (arg == "-tr" && i + 1 < argc) {
            thresh = std::stod(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -n <number of threads> -t <time interval [0,t]> -c <coarse timestep> -s <number of subcycles> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    T = T == 0 ? num_thr * coarse_dt : T;
    PRINT("Simulating on", num_thr, "cores with time interval = [0, ", T, "], coarse timestep =", coarse_dt, ", fine timestep =", fine_dt, "and", Nsub,"subcycles for the coarse solver.");
    
    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3,Nx), Bc(3,Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);
    int NT = round(T / coarse_dt);
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, coarse_dt, qp,LinSolvers::SolverType::GMRES);
    auto fine_solver = ECSIM<1, 3>(L, Np, fine_Nx, 1, fine_dt, qp, LinSolvers::SolverType::GMRES);
    auto para_solver = Parareal(fine_solver, coarse_solver, thresh, NT + 1, num_thr);

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn.col(0) << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    //coarse_solver.Step(Xn, 0, 10*coarse_dt, Xn); // Get into regime for E and B

    auto Eold = coarse_solver.Energy(Xn);
    VectorXd Yn(4 * Np + 6 * fine_Nx), fine_Xn(4 * Np + 6 * fine_Nx);
    VectorXd coarse_Yn(4 * Np + 6 * Nx);
    auto tic = std::chrono::high_resolution_clock::now();
    fine_solver.Refine(Xn, coarse_solver, fine_Xn);
    fine_solver.Step(fine_Xn, 0, T, Yn);
    coarse_solver.Coarsen(Yn, fine_solver, coarse_Yn);
    auto toc = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("SERIAL simulation takes", serial_time, "ms");
    PRINT("Serial energy difference between beginning and end", abs((coarse_solver.Energy(coarse_Yn) - Eold).sum()) / abs(Eold.sum()));

    PRINT("------PARAREAL------");
    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
    MatrixXd Xn_para(4 * Np + 6 * Nx, NT + 1);
    Xn_para.col(0) = Xn;
    VectorXd Ediff_para(NT);

    std::cout << std::setprecision(16);
    tic = std::chrono::high_resolution_clock::now();
    para_solver.Solve(Xn_para, ts);
    toc = std::chrono::high_resolution_clock::now();
    double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("PARAREAL simulation takes", para_time, "ms");
    PRINT("Speedup =", serial_time/para_time);
    PRINT("Energy difference parareal", abs((coarse_solver.Energy(Xn_para.col(NT)) - Eold).sum()) / abs(Eold.sum()));
    PRINT("Error in states of parareal compared to serial:", coarse_solver.Error(coarse_Yn, Xn_para.col(NT)));

    return 0;
}

#endif