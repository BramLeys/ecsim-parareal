#ifndef ODE_SOLVERS_H
#define ODE_SOLVERS_H
#include <Eigen/Dense>

// Eigen is column-major by default
using Eigen::MatrixXd;
using Eigen::VectorXd;


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

#endif