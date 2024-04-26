#ifndef SOLVER_TEST_3D_H
#define SOLVER_TEST_3D_H

#include "../ECSIM.h"
#include "../parareal.h"
#include "../common.h"

using namespace Eigen;

int solver_test(int argc, char* argv[]) {
    double L = 2 * EIGEN_PI; // Size of position space
    int Np = 10000; // number of particles
    int Nx = 512; // number of grid cells
    int fine_Nx = Nx;
    double coarse_dt = 1e-2;
    double fine_dt = 1e-4;
    int num_thr = 12;
    double T = 12 * coarse_dt;
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
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -c <coarse timestep> -n <number of threads> -t <time interval [0,t]> -nxc <coarse grid cells> -nxf <fine grid cells> -s <number of subcycles> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    PRINT("Simulating on", num_thr, "cores with time interval = [0, ", T, "], coarse timestep =", coarse_dt, ", fine timestep =", fine_dt, "and", Nsub, "subcycles for the coarse solver.");

    ArrayXd xp(Np), qp(Np);
    Array3Xd vp(3, Np), E0(3, Nx), Bc(3, Nx);

    TestProblems::SetTransverse(xp, vp, E0, Bc, qp, Nx, Np, L);
    auto coarse_solver = ECSIM<1, 3>(L, Np, Nx, Nsub, coarse_dt, qp, LinSolvers::SolverType::GMRES);
    auto fine_solver = ECSIM<1, 3>(L, Np, fine_Nx, 1, fine_dt, qp, LinSolvers::SolverType::LU);
    auto para_solver = Parareal(fine_solver, coarse_solver, thresh);
    int NT = T / coarse_dt;

    VectorXd Xn(4 * Np + 6 * Nx);
    Xn.col(0) << xp, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
    //coarse_solver.Step(Xn, 0, 10*coarse_dt, Xn); // Get into regime for E and B

    auto Eold = coarse_solver.Energy(Xn);
    VectorXd Yn(4 * Np + 6 * fine_Nx), fine_Xn(4 * Np + 6 * fine_Nx);
    VectorXd coarse_Yn(4 * Np + 6 * Nx);

    VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
    MatrixXd Xn_para(4 * Np + 6 * Nx, NT + 1);
    Xn_para.col(0) = Xn;
    VectorXd Ediff_para(NT);

    const int nb_solvers = 3;
    MatrixXd paralleltime(nb_solvers, nb_solvers);
    MatrixXd serialtime(nb_solvers, nb_solvers);
    MatrixXd iterations(nb_solvers, nb_solvers);
    std::cout << std::setprecision(16);
    std::string labels[nb_solvers] = { "LU", "GMRES", "BICGSTAB" };
    LinSolvers::SolverType types[nb_solvers] = { LinSolvers::SolverType::LU , LinSolvers::SolverType::GMRES, LinSolvers::SolverType::BICGSTAB};
    PRINT("LAYOUT:");
    PRINT("FINE SOLVER: LU\t GMRES\t BICGSTAB");
    PRINT("COARSE SOLVER:\n LU\n GMRES\n BICGSTAB");
    for (int i = 0; i < nb_solvers; i++) {
        PRINT("====================== FINE =", labels[i], " ======================");
        fine_solver.Set_solver(types[i]);
        auto tic = std::chrono::high_resolution_clock::now();
        fine_solver.Refine(Xn, coarse_solver, fine_Xn);
        fine_solver.Step(fine_Xn, 0, T, Yn);
        coarse_solver.Coarsen(Yn, fine_solver, coarse_Yn);
        auto toc = std::chrono::high_resolution_clock::now();
        double serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
        serialtime.col(i).setConstant(serial_time);
        PRINT("SERIAL simulation takes", serial_time, "ms");
        PRINT("Serial energy difference between beginning and end", abs((coarse_solver.Energy(coarse_Yn) - Eold).sum()) / abs(Eold.sum()));
        for (int j = 0; j < nb_solvers; j++) {
            PRINT("====================== COARSE =", labels[j], " ======================");
            coarse_solver.Set_solver(types[j]);
            tic = std::chrono::high_resolution_clock::now();
            iterations(j,i) = para_solver.Solve(Xn_para, ts);
            toc = std::chrono::high_resolution_clock::now();
            double para_time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
            PRINT("PARAREAL simulation takes", para_time, "ms");
            paralleltime(j, i) = para_time;
            PRINT("Speedup =", serial_time / para_time);
            PRINT("Energy difference parareal", abs((coarse_solver.Energy(Xn_para.col(NT)) - Eold).sum()) / abs(Eold.sum()));
            PRINT("Error in states of parareal compared to serial:", coarse_solver.Error(coarse_Yn, Xn_para.col(NT)));
        }
    }

    PRINT("SERIAL TIMES");
    PRINT(serialtime);
    PRINT("PARAREAL TIMES");
    PRINT(paralleltime);
    PRINT("SPEEDUP");
    PRINT(serialtime.array() / paralleltime.array());
    PRINT("ITERATIONS");
    PRINT(iterations);

    return 0;
}

#endif
