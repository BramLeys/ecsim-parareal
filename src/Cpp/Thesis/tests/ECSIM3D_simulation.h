#ifndef ECSIM_SIM3D_H
#define ECSIM_SIM3D_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int ECSIM3D_simulation(int argc, char* argv[]) {
    int Nx = 128; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    double dt = 0.5;
    double T = 62;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-dt" && i + 1 < argc) {
            dt = std::stod(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        }
        else if (arg == "-nx" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -dt <time step size> -t <time frame [0,t]> -nx <number of gridcells>" << std::endl;
            return 1;
        }
    }
    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);
    auto solver = ECSIM<1, 3>(L, Np, Nx, 1, dt, qp);
    int NT = round(T / dt);
    int dim = 4 * Np + 6 * Nx;

    VectorXd Xn(dim);
    Xn << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());

    auto Eold = solver.Energy(Xn);
    MatrixXd Yn(dim, NT + 1);
    Yn.col(0) = Xn;
    auto tic = std::chrono::high_resolution_clock::now();
    ArrayXd energies(NT + 1);
    energies(0) = Eold.sum();
    ArrayXd ts = ArrayXd::LinSpaced(NT + 1, 0, T);
    for (int i = 0; i < NT; i++) {
        solver.Step(Yn.col(i), ts(i), ts(i + 1), Yn.col(i + 1));
        energies(i + 1) = solver.Energy(Yn.col(i + 1)).sum();
        PRINT("Energy conservation for step", ts(i), "to ", ts(i + 1), "is equal to ", abs(energies(i + 1) - energies(0)) / energies(0));
    }
    auto toc = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Serial energy difference between beginning and end", abs((solver.Energy(Yn.col(NT)) - Eold).sum()) / abs(Eold.sum()));
    PRINT("Serial took", serial_time, "ms");
    save("simulation_result3D.txt", Yn(Eigen::all, seq(0, NT, NT)));
    save("simulation_energy3D.txt", energies);


    return 0;
}

#endif
