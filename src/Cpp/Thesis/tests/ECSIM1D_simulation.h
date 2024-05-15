#ifndef ECSIM_SIM1D_H
#define ECSIM_SIM1D_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int ECSIM1D_simulation(int argc, char* argv[]) {
    int Nx = 5000; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles


    double dt = 0.125;
    double T = 1;

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
    ArrayXd xp(Np), vp(Np), E0(Nx), Bc(Nx), qp(Np);

    TestProblems::SetTwoStream(xp, vp, E0, Bc, qp, Nx, Np, L);
    auto solver = ECSIM<1, 1>(L, Np, Nx, 1, dt, qp);
    int NT = round(T / dt);

    VectorXd Xn(2 * Np + 2 * Nx);
    Xn << xp, vp, E0, Bc;

    auto Eold = solver.Energy(Xn);
    MatrixXd Yn(2 * Np + 2 * Nx,NT + 1);
    Yn.col(0) = Xn;
    auto tic = std::chrono::high_resolution_clock::now();
    ArrayXd energies(NT+1);
    energies(0) = Eold.sum();
    ArrayXd ts = ArrayXd::LinSpaced(NT + 1, 0, T);
    for (int i = 0; i < NT; i++) {
        solver.Step(Yn.col(i), ts(i), ts(i+1), Yn.col(i + 1));
        energies(i+1) = solver.Energy(Yn.col(i + 1)).sum();
        PRINT("Energy conservation for step", i * dt, "to ", (i + 1) * dt, "is equal to ", abs(energies(i + 1) - energies(0)) / energies(0));
   }
    auto toc = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Serial energy difference between beginning and end", abs((solver.Energy(Yn.col(NT)) - Eold).sum()) / abs(Eold.sum()));
    PRINT("Serial took", serial_time, "ms");
    save("simulation_result1D.txt", Yn(Eigen::all,seq(0,NT,NT)));
    save("simulation_energy1D.txt", energies);


    return 0;
}

#endif
