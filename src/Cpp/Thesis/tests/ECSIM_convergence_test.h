#ifndef ECSIM_CONVERGENCE_TEST_3D_H
#define ECSIM_CONVERGENCE_TEST_3D_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <omp.h>

using namespace Eigen;

int ECSIM_convergence_test(int argc, char* argv[]) {
    int Nx = 128; // number of grid cells
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles


    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);

    //ArrayXd xp(Np), qp(Np), vp(Np), E0(Nx), Bc(Nx);
    //TestProblems::SetTwoStream(xp, vp, E0, Bc, qp, Nx, Np, L);

    double dt = 0.125;
    int num_thr = 12;
    double T = 0;
    int refinements = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-dt" && i + 1 < argc) {
            dt = std::stod(argv[++i]);
        }
        else if (arg == "-n" && i + 1 < argc) {
            num_thr = std::stoi(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        }
        else if (arg == "-r" && i + 1 < argc) {
            refinements = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -n <number of threads> -t <time interval [0,t]> -c <coarse timestep> -s <number of subcycles> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    T = T == 0 ? num_thr * dt : T;
    auto solver = ECSIM<1, 3>(L, Np, Nx, 1, dt, qp);
    std::cout << std::setprecision(16);
    int dimension = (solver.Get_xdim() + solver.Get_vdim()) * Np + 2 * solver.Get_vdim() * Nx;

    VectorXd Xn(dimension);
    Xn.col(0) << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());

    auto Eold = solver.Energy(Xn);
    VectorXd Yn_ref(dimension);
    solver.Set_dt(dt / pow(2, refinements));
    auto tic = std::chrono::high_resolution_clock::now();
    solver.Step(Xn, 0, T, Yn_ref);
    auto toc = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
    PRINT("REF simulation takes", serial_time, "ms");
    PRINT("REF energy difference between beginning and end", abs((solver.Energy(Yn_ref) - Eold).sum()) / abs(Eold.sum()));

    ArrayXXd errors(refinements,5);
    ArrayXXd convergence(refinements, 5);
    VectorXd Yn = VectorXd::Zero(dimension);
    for (int i = 0; i < refinements; i++) {
        solver.Set_dt(dt / pow(2,i));
        PRINT("DT = ", solver.Get_dt());
        tic = std::chrono::high_resolution_clock::now();
        solver.Step(Xn, 0, T, Yn);
        toc = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        PRINT("Simulation takes", time, "ms");
        PRINT("Energy difference sim", abs((solver.Energy(Yn) - Eold).sum()) / abs(Eold.sum()));
        PRINT("Shape yn:", Yn.rows(), "x", Yn.cols());
        errors(i, 0) = solver.Get_dt();
        errors.row(i).rightCols(4) = solver.Error(Yn, Yn_ref).transpose();
        PRINT("Error in states of sim compared to ref:", errors.row(i));
        if (i > 0) {
            convergence(i, 0) = solver.Get_dt();
            convergence.row(i).rightCols(4) = errors.row(i).rightCols(4) / errors.row(i - 1).rightCols(4);
        }
    }
    PRINT("ERRORS = ", errors);
    PRINT("CONVERGENCE = ", convergence);
    save("convergence_rate.txt", convergence);
    save("convergence_errors.txt", errors);
    return 0;
}

#endif