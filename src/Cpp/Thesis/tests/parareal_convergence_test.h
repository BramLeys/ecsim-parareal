#ifndef PARAREAL_CONVERGENCE_TEST_H
#define PARAREAL_CONVERGENCE_TEST_H

#include <Eigen/Dense>
#include "../solvers.h"
#include "../parareal.h"

using namespace Eigen;


int parareal_convergence_test(int argc, char* argv[])
{
	const int N = 2;
	VectorXd lambdas(N);
	lambdas << -1, -2;
	auto my_fun = [&lambdas](const Ref<const MatrixXd> xn, double tn, Ref<MatrixXd> yn) {yn = xn.cwiseProduct(-lambdas.cwiseAbs()); };
	auto analytical = [&lambdas](const Eigen::VectorXd& t) {
		// Vector case: return 2D array
		Eigen::MatrixXd result(lambdas.size(), t.size());
		for (int j = 0; j < t.size(); ++j) {
			result.col(j) = (lambdas.array() * t(j)).exp();
		}
		return result;
	};
	double T = 5;
	double coarse_dt = 1e-1;
	double fine_dt = coarse_dt / 10;


	int NT = (int)(T / coarse_dt);
	VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
	int refinement = 5;
	int iterations = 6;

	auto F = ForwardEuler<decltype(my_fun)>(my_fun, fine_dt);
	auto G = ForwardEuler<decltype(my_fun)>(my_fun, coarse_dt);
	auto parareal_solver = Parareal<decltype(F), decltype(G)>(F, G);

	MatrixXd X_para(N, NT + 1);
	MatrixXd X_fine(N, NT + 1);
	MatrixXd errors(2, refinement);
	VectorXd fine_mesh(refinement);
	X_para.col(0) = analytical(VectorXd::Constant(1,0.));
	X_fine.col(0) = X_para.col(0);
	for (int i = 0; i < refinement; i++) {
		fine_dt = coarse_dt / pow(10.,i);
		fine_mesh(i) = fine_dt;
		F.Set_dt(fine_dt);
		//X.col(0) = VectorXd::Random(N);
		parareal_solver.Solve(X_para, ts);
		errors(0,i) = (X_para - analytical(ts)).colwise().norm().cwiseQuotient(analytical(ts).colwise().norm()).norm();
		for (int j = 0; j < NT ; j++) {
			F.Step(X_fine.col(j), ts(j), ts(j+1), X_fine.col(j + 1));
		}
		errors(1,i) = (X_fine - analytical(ts)).colwise().norm().cwiseQuotient(analytical(ts).colwise().norm()).norm();
	}
	PRINT("Analytical solution: ", analytical(ts));
	PRINT("fine solution for dt =", fine_dt," : ", X_fine);
	PRINT("Errors: ", errors);
	save("Convergence_errors_parareal_vs_fine.mat", errors);
	save("Convergence_errors_parareal_vs_fine_dt.mat", fine_mesh);
	return 0;
}
#endif