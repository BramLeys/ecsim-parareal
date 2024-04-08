#ifndef LINEAR_SOLVERS_H
#define LINEAR_SOLVERS_H
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/IterativeSolvers>

namespace LinSolvers {
    enum class SolverType {
        BICGSTAB,
        GMRES,
        LU
    };
	class SparseSolverBase {
	public: 
		virtual inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const = 0;
	};

    class BICGSTABSolver : public SparseSolverBase {
    public:
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;
            solver.setMaxIterations(100);
            solver.setTolerance(1e-7);
            return solver.compute(A).solveWithGuess(b, initialGuess);
        }
    };

    class GMRESSolver : public SparseSolverBase {
    public:
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::GMRES<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;
            solver.setMaxIterations(100);
            solver.setTolerance(1e-7);
            return solver.compute(A).solveWithGuess(b, initialGuess);
        }
    };

    class LUSolver : public SparseSolverBase {
    public:
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.analyzePattern(A);
            solver.factorize(A);
            if (solver.info() != Eigen::Success) {
                std::cerr << "LU decomposition failed" << std::endl;
                // Return initial guess if LU decomposition fails
                return initialGuess;
            }
            return solver.solve(b);
        }
    };

    class LinSolver {
    private:
        std::unique_ptr<SparseSolverBase> solver_;
    public:
        LinSolver(LinSolvers::SolverType type) {
            switch (type) {
                case SolverType::GMRES:
                    solver_ = std::make_unique<LinSolvers::GMRESSolver>();
                    break;
                case SolverType::LU:
                    solver_ = std::make_unique < LinSolvers::LUSolver>();
                    break;
                case SolverType::BICGSTAB:
                    solver_ = std::make_unique < LinSolvers::BICGSTABSolver>();
                    break;
                default:
                    std::cerr << "Unkown solver, defaulting to LU Solver" << std::endl;
                    solver_ = std::make_unique < LinSolvers::LUSolver>();
            }

        }

        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const {
            return solver_->solve(A, b, initialGuess);
        }
    };
}

#endif
