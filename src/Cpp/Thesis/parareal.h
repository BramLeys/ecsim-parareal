#ifndef PARAREAL_SOLVERS_H
#define PARAREAL_SOLVERS_H
#include <Eigen/Dense>
#include <omp.h>
#include "debug.h"
#include "common.h"

using namespace Eigen;


template <typename F, typename G>
class Parareal {
private:
	F & fine;
	G & coarse;
	int it;
	double thresh;
	int num_threads;
public:
	Parareal(F & fine_solver, G & coarse_solver, double threshold = 1e-8, int max_iterations = 50, int num_thr = 12)
		:fine(fine_solver), coarse(coarse_solver), thresh(threshold), it(max_iterations),num_threads(num_thr)
	{}

	// T contains all timesteps at which states are found (initial conditions included -> N+1, where N is number of timesteps)
	// X contains initial condition in first col on entry and full simulation result on exit
	int Solve(MatrixXd& X, VectorXd& T, MatrixXd* ref=nullptr) {
		//auto Eold = coarse.Energy(X.col(0));
		// Perform coarse simulation
		for (Eigen::Index i = 0; i < T.size() - 1; i++) {
			coarse.Step(X.col(i), T(i), T(i + 1), X.col(i + 1));
		}
		Eigen::MatrixXd fine_x(X.rows(), T.size() - 1), coarse_x(X.rows(), T.size() - 1), new_coarse_x(X.rows(), 1);
		coarse_x = X.rightCols(coarse_x.cols());
		// keeps track of parareal iteration 
		int k = 0;
		// the index up to which (inclusive) the algorithm has converged
		int converged_until = 0;
		Eigen::MatrixXd previous_X(X.rows(), X.cols());
		Eigen::ArrayXXd diffs = ArrayXXd::Zero(it, 4);
		//save("Parareal_states_iteration_" + std::to_string(k) + ".txt", X);
		while ((k < it) && (converged_until < T.size() - 1)) {
			auto paratic = std::chrono::high_resolution_clock::now();
			k++;
			//auto tic = std::chrono::high_resolution_clock::now();
			#pragma omp parallel for num_threads(num_threads)
			for (int i = converged_until; i < T.size() - 1; i++) {
				fine.Step(X.col(i), T(i), T(i + 1), fine_x.col(i));
			}
			//auto toc = std::chrono::high_resolution_clock::now();
			//PRINT("Finished parallel section in ", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
			previous_X = X;
			for (Eigen::Index i = converged_until; i < T.size() - 1; i++) {
				coarse.Step(X.col(i), T(i), T(i + 1), new_coarse_x);
				X.col(i + 1) = new_coarse_x + fine_x.col(i) - coarse_x.col(i);
				coarse_x.col(i) = new_coarse_x;
				if ((converged_until == i) && (coarse.Error(X.col(i + 1), previous_X.col(i + 1)).maxCoeff() <= thresh)) {
					converged_until++;
				}
				diffs(k - 1, 0) = k;
				diffs(k - 1, 1) = std::max(diffs(k - 1, 1), coarse.Error(X.col(i + 1), previous_X.col(i + 1)).maxCoeff());
				if (ref != nullptr) {
					diffs(k - 1, 2) = std::max(diffs(k - 1, 2), coarse.Error(ref->col(i + 1), X.col(i + 1)).maxCoeff());
				}
				//diffs(k - 1, 3) = std::max(diffs(k - 1, 3), abs((coarse.Energy(X.col(i + 1)) - Eold).sum()) / abs(Eold.sum()));
			}
			//save("Parareal_states_iteration_" + std::to_string(k) + ".txt", X);
			auto paratoc = std::chrono::high_resolution_clock::now();
			PRINT("For iteration", k, ": time taken =", std::chrono::duration_cast<std::chrono::milliseconds>(paratoc - paratic).count(), "ms,	max state change = ", diffs(k - 1, 1), "	and time steps until and including", converged_until, "have converged");
			if (ref != nullptr) {
				PRINT("Actual error: ", diffs(k - 1, 2));
			}
			//PRINT("Energy conservation:", diffs(k - 1, 3));
			//if (diffs(k - 1, 1) < thresh){
			//	break;
			//}

		}
		PRINT("Parareal took", k, "iterations.");
		save("parareal_iteration_information.txt", diffs.topRows(k));
		return k;
	}
};

#endif