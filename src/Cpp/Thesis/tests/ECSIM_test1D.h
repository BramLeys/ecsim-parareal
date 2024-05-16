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

    double coarse_dt = 1e-2;
    double fine_dt = 1e-4;
    int num_thr = 12;
    double T = 0;
    double thr = 1e-8;
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
        else if (arg == "-tr" && i + 1 < argc) {
            thr = std::stod(argv[++i]);
        }
        else if (arg == "-nx" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <value> -n <value> -t <value> -c <value> -s <value> -tr <value> -nx <value>" << std::endl;
            return 1;
        }
    }
    T = T == 0 ? 12 * coarse_dt : T;
    
    ArrayXd xp(Np), vp(Np), E0(Nx), Bc(Nx),qp(Np);

    TestProblems::SetTwoStream(xp,vp,E0,Bc,qp,Nx,Np,L);
    auto coarse_solver = ECSIM<1, 1>(L, Np, Nx, Nsub, coarse_dt, qp);
    auto fine_solver = ECSIM<1, 1>(L, Np, Nx, 1, fine_dt, qp);
    auto para_solver = Parareal(fine_solver,coarse_solver,thr);
    int NT = T / coarse_dt;

    VectorXd Xn(2 * Np + 2 * Nx);
    Xn << xp, vp, E0, Bc;

    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
    auto Eold = fine_solver.Energy(Xn);
    MatrixXd Yn(2 * Np + 2 * Nx, NT + 1);
    Yn.col(0) = Xn;
    auto tic = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NT; i++)
        fine_solver.Step(Yn.col(i), ts(i), ts(i + 1), Yn.col(i + 1));
    auto toc = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Serial energy difference between beginning and end", abs((fine_solver.Energy(Yn) - Eold).sum()) / abs(Eold.sum()));
    PRINT("Serial took", serial_time, "ms");


    PRINT("------PARAREAL------");
    MatrixXd Xn_para(2 * Np + 2 * Nx, NT + 1);
    Xn_para.col(0) = Xn;
    VectorXd Ediff_para(NT);

    tic = std::chrono::high_resolution_clock::now();
    para_solver.Solve(Xn_para, ts, &Yn);
    toc = std::chrono::high_resolution_clock::now();
    double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Parareal took", para_time,"ms");
    PRINT("Parareal without subcycling takes", para_time/serial_time, "as long as the serial version");
    PRINT("Energy difference perturbed", abs((fine_solver.Energy(Xn_para.col(NT)) - Eold).sum()) / abs(Eold.sum()));
    PRINT("Diff in energy after simulation = ", abs((fine_solver.Energy(Yn) - fine_solver.Energy(Xn_para.col(NT))).sum()));

    return 0;
}

#endif
