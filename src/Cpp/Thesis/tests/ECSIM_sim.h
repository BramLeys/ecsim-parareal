#ifndef ECSIM_TEST_SIM_H
#define ECSIM_TEST_SIM_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int ECSIM_3D_sim(int argc, char* argv[]) {
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    int Nx = 128; // number of grid cells
    double dt = 0.125;
    double T = 62.5;
    int Nsub = 1;
    double thresh = 1e-8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-dt" && i + 1 < argc) {
            dt = std::stod(argv[++i]);
        }
        else if (arg == "-nx" && i + 1 < argc) {
            Nx = std::stoi(argv[++i]);
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
            std::cerr << "Usage: " << argv[0] << " -dt <timestep> -t <time interval [0,t]> -s <number of subcycles> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    PRINT("Simulating with time interval = [0, ", T, "] with timesteps =", dt, "and", Nsub, "subcycles.");

    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);
    int NT = round(T / dt);
    auto solver = ECSIM<1, 3>(L, Np, Nx, Nsub, dt, qp, LinSolvers::SolverType::LU);

    MatrixXd Xn(4 * Np + 6 * Nx, NT+1);
    ArrayXd energy(NT + 1);
    Xn.col(0) << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    energy(0) = solver.Energy(Xn.col(0)).sum();
    std::cout << std::setprecision(16);
    auto tic = std::chrono::high_resolution_clock::now();
    //#pragma omp parallel for
    for (int i = 0; i < NT; i++) {
        PRINT("Simulating step", i);
        solver.Step(Xn.col(i), i*dt, (i+1)*dt, Xn.col(i+1));
        energy(i + 1) = solver.Energy(Xn.col(i + 1)).sum();
    }
    auto toc = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("Simulation of ",NT,"timesteps took", time, "ms");
    PRINT("Energy conservation = ", abs(energy(NT) - energy(0)) / energy(0));

    save("ECSIM_simulation.txt", Xn);
    save("ECSIM_energy.txt", energy);

    return 0;
}

#endif