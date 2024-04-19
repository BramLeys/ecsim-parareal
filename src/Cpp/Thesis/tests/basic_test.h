#ifndef BASIC_TEST_H
#define BASIC_TEST_H

#include <Eigen/Dense>
#include "../solvers.h"
#include "../parareal.h"

using namespace Eigen;


int basic_test(int argc, char* argv[]){
	int N = 10;
	double T = 0;
	double coarse_dt = 1e-2;
	double fine_dt = 1e-4;
    double thresh = 1e-8;
    double L = 2 * EIGEN_PI;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" && i + 1 < argc) {
            fine_dt = std::stod(argv[++i]);
        }
        else if (arg == "-c" && i + 1 < argc) {
            coarse_dt = std::stod(argv[++i]);
        }
        else if (arg == "-t" && i + 1 < argc) {
            T = std::stod(argv[++i]);
        }
        else if (arg == "-tr" && i + 1 < argc) {
            thresh = std::stod(argv[++i]);
        }
        else if (arg == "-N" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Usage: " << argv[0] << " -f <fine timestep> -t <time interval [0,t]> -c <coarse timestep> -tr <parareal threshold>" << std::endl;
            return 1;
        }
    }
    T = T == 0 ? 12 * coarse_dt : T;
    double dx = L / N;
    auto second_order = [&dx](const Ref<const MatrixXd> xn, double tn, Ref<MatrixXd> yn) {
        for (int j = 0; j < yn.cols(); j++) {
            for (int i = 0; i < yn.rows(); i++) {
                yn(i, j) = (xn((i - 1 + yn.rows()) % yn.rows(), j) - 2 * xn(i, j) + xn((i + 1) % yn.rows(), j)) / (dx*dx);
            }
        }
    };
    auto first_order = [&dx](const Ref<const MatrixXd> xn, double tn, Ref<MatrixXd> yn) {
        for (int j = 0; j < yn.cols(); j++) {
            for (int i = 0; i < yn.rows(); i++) {
                yn(i, j) = (-xn((i - 1 + yn.rows()) % yn.rows(), j) + xn((i + 1) % yn.rows(), j)) / (2*dx);
            }
        }
    };
	auto analytical = [&L,&dx](int N, double t) { 
        return (Eigen::ArrayXd::LinSpaced(N,0,L-dx) + t).sin();
    };

    Eigen::SparseMatrix<double> A(N, N);
    A.reserve(Eigen::VectorXi::Constant(N, 2));
    for (int i = 0; i < N; ++i) {
        A.insert(i, (i + 1) % N) = 1 / (2 * dx);
        A.insert(i, (i - 1 + N) % N) = -1 / (2 * dx);
    }
    A.makeCompressed();

	int NT = (int)(T / coarse_dt);

	auto fine_solver = CrankNicolson(A, fine_dt);
	auto coarse_solver = CrankNicolson(A, coarse_dt);
	auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver, thresh);

	MatrixXd X(N, NT + 1);
	X.col(0) = VectorXd::Random(N);
	VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
	parareal_solver.Solve(X, ts);
    MatrixXd X_serial(N, NT + 1);
    X_serial.col(0) = X.col(0);
    for (int i = 1; i < NT+1; i++) {
        fine_solver.Step(X_serial.col(i - 1), ts[i - 1], ts[i], X_serial.col(i));
    }
    PRINT("coarse_dt/dx = ", coarse_dt / dx);
    PRINT("fine_dt/dx = ", fine_dt / dx);
    PRINT("Max parareal vs serial error = ", fine_solver.Error(X, X_serial).maxCoeff());
	return 0;
}

#endif