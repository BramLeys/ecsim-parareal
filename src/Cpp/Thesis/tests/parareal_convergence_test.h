#ifndef PARAREAL_CONVERGENCE_TEST_H
#define PARAREAL_CONVERGENCE_TEST_H

#include <Eigen/Dense>
#include "../solvers.h"
#include "../parareal.h"

using namespace Eigen;


int parareal_convergence_test(int argc, char* argv[])
{
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
	int refinement = 5;

	auto ref_solver = CrankNicolson(B, coarse_dt/pow(2,refinement+3));
	auto F = CrankNicolson(B, fine_dt);
	auto G = CrankNicolson(B, coarse_dt);
	//auto ref_solver = RK4<decltype(second_order)>(second_order, coarse_dt/pow(2,refinement+3));
	//auto F = RK4<decltype(second_order)>(second_order,fine_dt);
	//auto G = RK4<decltype(second_order)>(second_order,coarse_dt);

	auto parareal_solver = Parareal<decltype(F), decltype(G)>(F, G, thresh);

	VectorXd Xn = (2 * ArrayXd::LinSpaced(N,0, L-dx)).sin() + 5;
	//VectorXd Xn = VectorXd::Random(N);
	MatrixXd X_para(N, NT + 1);
	VectorXd Yn_ref(N), Yn_ser(N);
	ref_solver.Step(Xn, 0, T, Yn_ref);
	ArrayXXd errors(refinement,3);
	ArrayXXd convergence(refinement, 3);
	X_para.col(0) = Xn;
	for (int i = 0; i < refinement; i++) {
		fine_dt = coarse_dt / pow(2,i);
		F.Set_dt(fine_dt);
		parareal_solver.Solve(X_para, ts);
		F.Step(Xn, 0, T, Yn_ser);
		errors(i, 0) = fine_dt;
		errors(i,1) = (X_para.col(NT) - Yn_ref).norm()/Yn_ref.norm();
		errors(i,2) = (Yn_ser - Yn_ref).norm()/Yn_ref.norm();
		if (i > 0) {
			convergence(i, 0) = fine_dt;
			convergence.row(i).rightCols(2) = errors.row(i).rightCols(2) / errors.row(i - 1).rightCols(2);
			PRINT("Parareal convergence = ", convergence(i, 1));
			PRINT("Serial convergence = ", convergence(i, 2));
		}
	}
	PRINT("coarse_dt/dx^2 = ", coarse_dt / dx / dx);
	PRINT("Errors:", errors);
	PRINT("Convergence:", convergence);
	save("convergence_errors.txt", errors);
	save("convergence_rate.txt", convergence);
	return 0;
}
#endif