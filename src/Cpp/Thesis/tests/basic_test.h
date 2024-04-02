#ifndef BASIC_TEST_H
#define BASIC_TEST_H

#include <Eigen/Dense>
#include "../solvers.h"
#include "../parareal.h"

using namespace Eigen;


int basic_test(int argc, char* argv[]){
	const int N = 2;
	VectorXd lambdas(N);
	lambdas << -1, -2;
	auto my_fun = [&lambdas](const Ref<const MatrixXd> xn, double tn, Ref<MatrixXd> yn) {yn = xn.cwiseProduct(-lambdas.cwiseAbs()); };
	auto analytical = [&lambdas](double t) { return (lambdas.array() * t).exp(); };
	double T = 10;
	double coarse_dt = 1e-1;
	double fine_dt = coarse_dt / 10;

	int NT = (int)(T / coarse_dt);

	auto fine_solver = ForwardEuler<decltype(my_fun)>(my_fun, fine_dt);
	auto coarse_solver = ForwardEuler<decltype(my_fun)>(my_fun, coarse_dt);
	auto parareal_solver = Parareal<decltype(fine_solver), decltype(coarse_solver)>(fine_solver, coarse_solver);

	MatrixXd X(N, NT + 1);
	//X.col(0) = VectorXd::Random(N);
	X.col(0) << 10, 10;
	VectorXd ts = VectorXd::LinSpaced(NT + 1, 0, T);
	parareal_solver.Solve(X, ts);
	return 0;
}

#endif