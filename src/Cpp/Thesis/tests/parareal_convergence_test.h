#ifndef PARAREAL_CONVERGENCE_TEST_H
#define PARAREAL_CONVERGENCE_TEST_H

#include <Eigen/Dense>
#include "../solvers.h"
#include "../parareal.h"

using namespace Eigen;


int parareal_convergence_test(int argc, char* argv[])
{
	int N = 100;
	double T = 0;
	double coarse_dt = 1e-2;
	double fine_dt = 1e-4;
	double thresh = 1e-8;
	double L = 2 * EIGEN_PI;
	int refinement = 5;
	std::string initialisation = "smooth";

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "-c" && i + 1 < argc) {
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
		else if (arg == "-r" && i + 1 < argc) {
			refinement = std::stoi(argv[++i]);
		}
		else if (arg == "-init" && i + 1 < argc) {
			initialisation = argv[++i];
			if (!(initialisation == "random" || initialisation == "smooth")) {
				std::cerr << "Usage: " << argv[0] << " -t <time interval [0,t]> -c <coarse timestep> -tr <parareal threshold> -N <number of gridcells> -r <number of refinements> -init <'random'|'smooth'>" << std::endl;
				return 1;
			}
		}
		else {
			std::cerr << "Usage: " << argv[0] << " -t <time interval [0,t]> -c <coarse timestep> -tr <parareal threshold> -N <number of gridcells> -r <number of refinements> -init <'random'|'smooth'>" << std::endl;
			return 1;
		}
	}
	T = T == 0 ? 12 * coarse_dt : T;

	double dx = L / N;
	auto second_order = [&dx](const Ref<const MatrixXd> xn, double tn, Ref<MatrixXd> yn) {
		for (int j = 0; j < yn.cols(); j++) {
			for (int i = 0; i < yn.rows(); i++) {
				yn(i, j) = (xn((i - 1 + yn.rows()) % yn.rows(), j) - 2 * xn(i, j) + xn((i + 1) % yn.rows(), j)) / (dx * dx);
			}
		}
		};
	auto first_order = [&dx](const Ref<const MatrixXd> xn, double tn, Ref<MatrixXd> yn) {
		for (int j = 0; j < yn.cols(); j++) {
			for (int i = 0; i < yn.rows(); i++) {
				yn(i, j) = (-xn((i - 1 + yn.rows()) % yn.rows(), j) + xn((i + 1) % yn.rows(), j)) / (2 * dx);
			}
		}
		};

	Eigen::SparseMatrix<double> A(N, N);
	A.reserve(Eigen::VectorXi::Constant(N, 2));
	for (int i = 0; i < N; ++i) {
		A.insert(i, (i + 1) % N) = 1 / (2 * dx);
		A.insert(i, (i - 1 + N) % N) = -1 / (2 * dx);
	}
	A.makeCompressed();

	Eigen::SparseMatrix<double> B(N, N);
	B.reserve(Eigen::VectorXi::Constant(N, 3));
	for (int i = 0; i < N; ++i) {
		B.insert(i, (i + 1) % N) = 1 / (dx * dx);
		B.insert(i, i) = -2 / (dx * dx);
		B.insert(i, (i - 1 + N) % N) = 1 / (dx * dx);
	}
	B.makeCompressed();


	int NT = (int)(T / coarse_dt);
	VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);

	auto ref_solver = CrankNicolson(B, coarse_dt / pow(2, refinement-1)/50, 1e-15);
	auto F = CrankNicolson(B, fine_dt, thresh / 100);
	auto G = CrankNicolson(B, coarse_dt, thresh / 100);
	//auto ref_solver = RK4<decltype(second_order)>(second_order, coarse_dt/pow(2,refinement+3));
	//auto F = RK4<decltype(second_order)>(second_order,fine_dt);
	//auto G = RK4<decltype(second_order)>(second_order,coarse_dt);

	auto parareal_solver = Parareal<decltype(F), decltype(G)>(F, G, thresh);

	VectorXd Xn = VectorXd(N);
	if (initialisation == "random")
		Xn = VectorXd::Random(N);
	else
		Xn = (2 * ArrayXd::LinSpaced(N, 0, L - dx-1)).sin() + 5;
	MatrixXd X_para(N, NT + 1);
	VectorXd Yn_ref(N);
	MatrixXd Yn_ser(N, NT + 1);
	ref_solver.Step(Xn, 0, T, Yn_ref);
	PRINT("Reference norm:", Yn_ref.norm());
	ArrayXXd errors = ArrayXXd::Zero(refinement, 3);
	ArrayXXd convergence = ArrayXXd::Zero(refinement, 3);
	X_para.col(0) = Xn;
	Yn_ser.col(0) = Xn;
	for (int i = 0; i < refinement; i++) {
		fine_dt = coarse_dt / pow(2, i);
		F.Set_dt(fine_dt);
		for(int j = 0; j < NT; j++)
			F.Step(Yn_ser.col(j), ts(j), ts(j+1), Yn_ser.col(j+1));
		parareal_solver.Solve(X_para, ts, &Yn_ser);
		errors(i, 0) = fine_dt;
		errors(i, 1) = (X_para.col(NT) - Yn_ref).norm() / Yn_ref.norm();
		errors(i, 2) = (Yn_ser.col(NT) - Yn_ref).norm() / Yn_ref.norm();
		if (i > 0) {
			convergence(i, 0) = fine_dt;
			convergence.row(i).rightCols(2) = errors.row(i).rightCols(2) / errors.row(i - 1).rightCols(2);
		}
	}
	PRINT("coarse_dt/dx^2 = ", coarse_dt / dx / dx);
	PRINT("Errors:", errors);
	PRINT("Convergence:", convergence);
	save("cn_convergence_errors.txt", errors);
	return 0;
}
#endif