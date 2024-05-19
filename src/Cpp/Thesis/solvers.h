#ifndef ODE_SOLVERS_H
#define ODE_SOLVERS_H
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>
#include <Eigen/Eigenvalues>
#include "debug.h"

// Eigen is column-major by default
using namespace Eigen;


template <typename F>
class ForwardEuler{
private:
	double dt;
	F const& func;
public:
	ForwardEuler(F const& function, double dt)
		:func(function),dt(dt)
	{}

	double Get_dt() const { return dt; }
	void Set_dt(double new_dt) { dt = new_dt; }


	// on exit yn will contain full simulation from t0 to t1 with timestep defined by dt 
	void Solve(const Eigen::Ref<const Eigen::MatrixXd> xn, double t0, double t1, Eigen::Ref<Eigen::MatrixXd> yn) const{
		VectorXd ts = VectorXd::LinSpaced(yn.cols(), t0, t1);
		yn.col(0) = xn;
		for (Eigen::Index i = 0; i < ts.size() - 1; i++) {
			func(yn.col(i), ts(i), yn.col(i + 1));
			yn.col(i + 1) *= dt;
			yn.col(i + 1) += yn.col(i);
		}
	}
	// on exit yn will only contain the calculated state at t1, mod(t0-t1,dt) should be 0
	void Step(const Eigen::Ref<const Eigen::MatrixXd> xn, double t0, double t1,Eigen::Ref<Eigen::MatrixXd> yn) const {
		int nb_steps = (int)round(abs(t1 - t0) / dt);
		MatrixXd steps(yn.rows(), nb_steps + 1);
		Solve(xn, t0, t1, steps);
		// if interpolation factor is not a whole integer, then the method breaks down to first order
		yn = steps.col(steps.cols() - 1);
	}
};

template <typename F>
class RK4 {
private:
	double dt;
	F const& func;
public:
	RK4(F const& function, double dt)
		:func(function), dt(dt)
	{}

	double Get_dt() const { return dt; }
	void Set_dt(double new_dt) { dt = new_dt; }

	inline Eigen::ArrayXd Error(const Eigen::Ref<const MatrixXd> X, const Eigen::Ref<const MatrixXd> Y) const {
		Eigen::ArrayXd errors = Eigen::ArrayXd::Zero(X.cols());
		for (int i = 0; i < X.cols(); i++) {
			errors(i) = (X.col(i) - Y.col(i)).norm() / X.col(i).norm();
		}
		return errors;
	}

	// on exit yn will contain full simulation from t0 to t1 with timestep defined by dt 
	void Solve(const Eigen::Ref<const Eigen::MatrixXd> xn, double t0, double t1, Eigen::Ref<Eigen::MatrixXd> yn) const {
		VectorXd ts = VectorXd::LinSpaced(yn.cols(), t0, t1);
		yn.col(0) = xn;
		MatrixXd k(yn.rows(), 4);
		for (Eigen::Index i = 0; i < ts.size()-1; i++) {
			func(yn.col(i), ts(i), k.col(0));
			func(yn.col(i) + dt * k.col(0) / 2, ts(i) + dt / 2, k.col(1));
			func(yn.col(i) + dt * k.col(1) / 2, ts(i) + dt / 2, k.col(2));
			func(yn.col(i) + dt * k.col(2), ts(i) + dt, k.col(3));
			yn.col(i + 1) = yn.col(i) + dt * (k.col(0) + 2 * k.col(1) + 2 * k.col(2) + k.col(3)) / 6;
		}
	}
	// on exit yn will only contain the calculated state at t1, mod(t0-t1,dt) should be 0
	void Step(const Eigen::Ref<const Eigen::MatrixXd> xn, double t0, double t1, Eigen::Ref<Eigen::MatrixXd> yn) const {
		int nb_steps = (int)round(abs(t1 - t0) / dt);
		MatrixXd steps(yn.rows(), nb_steps + 1);
		Solve(xn, t0, t1, steps);
		// if interpolation factor is not a whole integer, then the method breaks down to first order
		yn = steps.col(nb_steps);
	}
};

class CrankNicolson {
private:
	double dt;
	const SparseMatrix<double>& A;
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IdentityPreconditioner> solver;
	//SparseLU<SparseMatrix<double>> solver;

public:
	CrankNicolson(const SparseMatrix<double>& A, double dt, double tol=1e-12)
		:A(A), dt(dt)
	{
		SparseMatrix<double> I(A.rows(), A.cols());
		I.setIdentity();
		solver.setTolerance(tol);
		solver.setMaxIterations(100);
		solver.compute(I - dt * 0.5 * A);
		//Eigen::EigenSolver<Eigen::MatrixXd> eigensolver;
		//eigensolver.compute((I - dt * 0.5 * A).toDense());
		//Eigen::VectorXd eigen_values = eigensolver.eigenvalues().real();
		//PRINT("eigenvalues: ", eigen_values);
	}

	double Get_dt() const { return dt; }
	void Set_dt(double new_dt) { 
		dt = new_dt;
		SparseMatrix<double> I(A.rows(), A.cols());
		I.setIdentity();
		solver.compute(I - dt * 0.5 * A);
		//Eigen::EigenSolver<Eigen::MatrixXd> eigensolver;
		//eigensolver.compute((I - dt * 0.5 * A).toDense());
		//Eigen::VectorXd eigen_values = eigensolver.eigenvalues().real();
		//PRINT("eigenvalues: ", eigen_values);

	}

	inline Eigen::ArrayXd Error(const Eigen::Ref<const MatrixXd> X, const Eigen::Ref<const MatrixXd> Y) const {
		Eigen::ArrayXd errors = Eigen::ArrayXd::Zero(X.cols());
		for (int i = 0; i < X.cols(); i++) {
			errors(i) = (X.col(i) - Y.col(i)).norm() / X.col(i).norm();
		}
		return errors;
	}

	// on exit yn will contain full simulation from t0 to t1 with timestep defined by dt 
	void Solve(const Eigen::Ref<const Eigen::MatrixXd> xn, double t0, double t1, Eigen::Ref<Eigen::MatrixXd> yn) const {
		yn.col(0) = xn;
		for (Eigen::Index i = 0; i < yn.cols()-1; i++) {
			yn.col(i + 1) = solver.solve(yn.col(i) + 0.5 * dt * A * yn.col(i));
		}
	}
	// on exit yn will only contain the calculated state at t1, mod(t0-t1,dt) should be 0
	void Step(const Eigen::Ref<const Eigen::MatrixXd> xn, double t0, double t1, Eigen::Ref<Eigen::MatrixXd> yn) const {
		int nb_steps = (int)round(abs(t1 - t0) / dt);
		MatrixXd steps(yn.rows(), nb_steps + 1);
		Solve(xn, t0, t1, steps);
		// if interpolation factor is not a whole integer, then the method breaks down to first order
		yn = steps.col(nb_steps);
	}
};

#endif