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
    int Nx = 128; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles

    double coarse_dt = 1e-2;
    double fine_dt = 1e-4;
    int num_thr = 12;
    double T = 12 * coarse_dt;
    int Nsub = 10;
    double thresh = 1e-8;

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
        else if (arg == "-nx" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        }
        else if (arg == "-tr" && i + 1 < argc) {
            thresh = std::stod(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -nx <number of gridcells> -n <number of threads> -t <time interval [0,t]> -c <coarse timestep> -s <number of subcycles> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    PRINT("Simulating on", num_thr, "cores with time interval = [0, ", T, "], coarse timestep =", coarse_dt, ", fine timestep =", fine_dt, "and", Nsub,"subcycles for the coarse solver.");
    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3,Nx), Bc(3,Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, 1, coarse_dt, qp);
    auto fine_solver = ECSIM<1, 3>(L, Np, Nx, 1, fine_dt, qp);
    auto para_solver = Parareal(fine_solver, coarse_solver,thresh);
    int NT = T / coarse_dt;

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn.col(0) << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    //coarse_solver.Step(Xn, 0, 10*coarse_dt, Xn); // Get into regime for E and B

    PRINT("Initial max divergence", abs(fine_solver.Divergence(Xn.col(0))).maxCoeff());

    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
    auto Eold = fine_solver.Energy(Xn);
    MatrixXd Yn(4 * Np + 6 * Nx, NT + 1);
    Yn.col(0) = Xn;
    auto tic = std::chrono::high_resolution_clock::now();
    for (int i = 0; i< NT; i++)
        fine_solver.Step(Yn.col(i), ts(i), ts(i + 1), Yn.col(i+1));
    auto toc = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("SERIAL simulation takes", serial_time, "ms");
    PRINT("Serial energy difference between beginning and end", abs((fine_solver.Energy(Yn.col(NT)) - Eold).sum()) / abs(Eold.sum()));

    PRINT("------PARAREAL------");
    MatrixXd Xn_para(4 * Np + 6 * Nx, NT + 1);
    Xn_para.col(0) = Xn;
    VectorXd Ediff_para(NT);

    std::cout << std::setprecision(16);
    tic = std::chrono::high_resolution_clock::now();
    para_solver.Solve(Xn_para, ts,&Yn);
    toc = std::chrono::high_resolution_clock::now();
    double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("PARAREAL simulation takes", para_time, "ms");
    PRINT("Parareal without subcycling takes", para_time / serial_time, "as long as the serial version");
    PRINT("Energy difference parareal", abs((fine_solver.Energy(Xn_para.col(NT)) - Eold).sum()) / abs(Eold.sum()));
    PRINT("Error in states of parareal compared to serial:", fine_solver.Error(Yn.col(NT), Xn_para.col(NT)));
    PRINT("ELECTRIC FIELD:", Xn_para.col(NT)(seqN(4*Np,3*Nx)));
    PRINT("MAGNETIC FIELD:", Xn_para.col(NT)(seqN(4 * Np + 3*Nx, 3 * Nx)));

    //auto temp = fine_solver.Divergence(Xn_para);
    //for (int i = 0; i < temp.cols();i++) {
    //    PRINT("Max parareal divergence of timestep",i," = ", abs(temp.col(i)).maxCoeff());
    //}
    //PRINT("Max serial divergence", abs(fine_solver.Divergence(Yn)).maxCoeff());
    return 0;
}

#endif