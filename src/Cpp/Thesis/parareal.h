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
	F const& fine;
	G const& coarse;
	int it;
	double thresh;
	int num_threads;
public:
	Parareal(F const& fine_solver, G const& coarse_solver, double threshold = 1e-8, int max_iterations = 50, int num_thr = 12)
		:fine(fine_solver), coarse(coarse_solver), thresh(threshold), it(max_iterations),num_threads(num_thr)
	{}

	void setNumThreads(int nb_threads) { num_threads = nb_threads; }

	// T contains all timesteps at which states are found (initial conditions included -> N+1, where N is number of timesteps)
	// X contains initial condition in first col on entry and full simulation result on exit
	int Solve(MatrixXd& X, VectorXd& T) {
		// Perform coarse simulation
		//auto tic = std::chrono::high_resolution_clock::now();
		for (Eigen::Index i = 0; i < T.size()-1; i++) {
			coarse.Step(X.col(i), T(i),T(i+1), X.col(i+ 1));
		}
		//auto toc = std::chrono::high_resolution_clock::now();
		//PRINT("Finished first coarse simulation in ", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
		int fine_dim = (fine.Get_xdim() + fine.Get_vdim()) * fine.Get_Np() + (2 * fine.Get_vdim()) * fine.Get_Nx();
		int coarse_dim = (coarse.Get_xdim() + coarse.Get_vdim()) * coarse.Get_Np() + (2 * coarse.Get_vdim()) * coarse.Get_Nx();
		Eigen::MatrixXd fine_x(fine_dim, T.size() - 1), coarse_x(coarse_dim, T.size() - 1), new_coarse_x(coarse_dim, 1);
		Eigen::MatrixXd coarsened_fine_x(coarse_dim, 1);
		coarse_x = X.rightCols(coarse_x.cols());

		//auto Etot0 = coarse.Energy(X.col(0)).sum();

		// keeps track of parareal iteration 
		int k = 0;
		// the index up to which(inclusive) the algorithm has converged
		int converged_until = 0;
		// Don't know if faster to copy one col each time or immediately copy full X matrix in temp
		Eigen::MatrixXd previous_X(X.rows(),X.cols());
		Eigen::Array<double,Dynamic,5> diffs = Eigen::Array<double, Dynamic, 5>::Zero(it, 5);
		//save("Parareal_states_iteration_" + std::to_string(k) + ".txt", X);
		while ((k < it) && (converged_until < T.size()-1)) {
			auto paratic = std::chrono::high_resolution_clock::now();
			k++;
			//tic = std::chrono::high_resolution_clock::now();
			#pragma omp parallel for num_threads(num_threads)
			for (int i = converged_until; i < T.size() - 1; i++) {
				Eigen::MatrixXd refined_coarse_x(fine_dim, 1);
				fine.Refine(X.col(i), coarse, refined_coarse_x);
				fine.Step(refined_coarse_x, T(i), T(i + 1), fine_x.col(i));
			}
			//toc = std::chrono::high_resolution_clock::now();
			//PRINT("Finished parallel section in ", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms");
			previous_X = X;
			//double coarse_time = 0;
			for (Eigen::Index i = converged_until; i < T.size() - 1; i++) {
				//tic = std::chrono::high_resolution_clock::now();
				coarse.Step(X.col(i), T(i), T(i + 1), new_coarse_x);
				coarse.Coarsen(fine_x.col(i), fine, coarsened_fine_x);
				//toc = std::chrono::high_resolution_clock::now();
				//coarse_time += std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count();
				X.col(i + 1) = coarsened_fine_x + new_coarse_x - coarse_x.col(i);
				coarse_x.col(i) = new_coarse_x;
				if ((converged_until == i) && (coarse.Error(X.col(i + 1), previous_X.col(i + 1)).maxCoeff() <= thresh)) {
					converged_until++;
				}
				diffs(k-1, 0) = k;
				diffs.row(k-1).rightCols(4) = diffs.row(k - 1).rightCols(4).max(coarse.Error(X.col(i + 1), previous_X.col(i + 1)).transpose());
			}
			//PRINT("Finished Coarse simulation in ", coarse_time, "ms");
			//save("Parareal_states_iteration_" + std::to_string(k) + ".txt", X);
			auto paratoc = std::chrono::high_resolution_clock::now();
			PRINT("For iteration",k,": time taken =", std::chrono::duration_cast<std::chrono::milliseconds>(paratoc - paratic).count(),"ms,	max state change = ", diffs.row(k - 1).rightCols(4).maxCoeff(),"	and time steps until and including", converged_until, "have converged");
			//PRINT("Energy conservation first and last step = ", abs(coarse.Energy(X.col(X.cols() - 1)).sum() - Etot0) / Etot0);

		}
		PRINT("Parareal took", k, "iterations.");
		save("parareal_iteration_information.txt", diffs.topRows(k));
		return k;
	}
};

#endif