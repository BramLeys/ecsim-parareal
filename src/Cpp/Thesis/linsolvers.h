#ifndef LINEAR_SOLVERS_H
#define LINEAR_SOLVERS_H
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <unsupported/Eigen/IterativeSolvers>

namespace LinSolvers {
    enum class SolverType {
        BICGSTAB,
        GMRES,
        LU,
        QR,
        SimplicialLDLT,
        SimplicialLLT,
        CG,
        LSCG,
        MINRES
    };
	class SparseSolverBase {
    protected:
        double thresh;
	public: 
        SparseSolverBase(double threshold=1e-10)
            :thresh(threshold)
        {}
		virtual inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const = 0;
	};

    class BICGSTABSolver : public SparseSolverBase {
    public:
        using SparseSolverBase::SparseSolverBase;
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IdentityPreconditioner> solver;
            solver.setMaxIterations(100);
            solver.setTolerance(this->thresh);
            return solver.compute(A).solveWithGuess(b, initialGuess);
        }
    };

    class GMRESSolver : public SparseSolverBase {
    public:
        using SparseSolverBase::SparseSolverBase;
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::GMRES<Eigen::SparseMatrix<double>, Eigen::IdentityPreconditioner> solver;
            solver.setMaxIterations(100);
            solver.setTolerance(this->thresh);
            return solver.compute(A).solveWithGuess(b, initialGuess);
        }
    };

    class MINRESSolver : public SparseSolverBase {
    public:
        using SparseSolverBase::SparseSolverBase;
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::MINRES<Eigen::SparseMatrix<double>> solver;
            solver.setMaxIterations(100);
            solver.setTolerance(this->thresh);
            return solver.compute(A).solveWithGuess(b, initialGuess);
        }
    };

    class LUSolver : public SparseSolverBase {
    public:
        using SparseSolverBase::SparseSolverBase;
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                std::cerr << "LU decomposition failed" << std::endl;
                // Return initial guess if LU decomposition fails
                return initialGuess;
            }
            return solver.solve(b);
        }
    };

    class QRSolver : public SparseSolverBase {
    public:
        using SparseSolverBase::SparseSolverBase;
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                std::cerr << "QR decomposition failed" << std::endl;
                // Return initial guess if QR decomposition fails
                return initialGuess;
            }
            return solver.solve(b);
        }
    };

    class LDLTSolver : public SparseSolverBase {
    public:
        using SparseSolverBase::SparseSolverBase;
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                std::cerr << "SimplicialLLT decomposition failed" << std::endl;
                // Return initial guess if SimplicialLLT decomposition fails
                return initialGuess;
            }
            return solver.solve(b);
        }
    };
    class LLTSolver : public SparseSolverBase {
    public:
        using SparseSolverBase::SparseSolverBase;
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                std::cerr << "SimplicialLDLT decomposition failed" << std::endl;
                // Return initial guess if SimplicialLDLT decomposition fails
                return initialGuess;
            }
            return solver.solve(b);
        }
    };

    class CGSolver : public SparseSolverBase {
    public:
        using SparseSolverBase::SparseSolverBase;
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                std::cerr << "Conjugate Gradient decomposition failed" << std::endl;
                // Return initial guess if Conjugate Gradient decomposition fails
                return initialGuess;
            }
            return solver.solve(b);
        }
    };

    class LSCGSolver : public SparseSolverBase {
    public:
        using SparseSolverBase::SparseSolverBase;
        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const override {
            Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>, Eigen::IdentityPreconditioner> solver;
            solver.compute(A);
            if (solver.info() != Eigen::Success) {
                std::cerr << "Least Squares Conjugate Gradient decomposition failed" << std::endl;
                // Return initial guess if Least Squares Conjugate Gradient decomposition fails
                return initialGuess;
            }
            return solver.solve(b);
        }
    };

    class LinSolver {
    private:
        std::unique_ptr<SparseSolverBase> solver_;
    public:
        LinSolver(LinSolvers::SolverType type, double threshold=1e-10) {
            switch (type) {
                case SolverType::GMRES:
                    solver_ = std::make_unique<LinSolvers::GMRESSolver>(threshold);
                    break;
                case SolverType::MINRES:
                    solver_ = std::make_unique<LinSolvers::MINRESSolver>(threshold);
                    break;
                case SolverType::LU:
                    solver_ = std::make_unique < LinSolvers::LUSolver>(threshold);
                    break;
                case SolverType::QR:
                    solver_ = std::make_unique<LinSolvers::QRSolver>(threshold);
                    break;
                case SolverType::SimplicialLDLT:
                    solver_ = std::make_unique<LinSolvers::LDLTSolver>(threshold);
                    break;
                case SolverType::SimplicialLLT:
                    solver_ = std::make_unique<LinSolvers::LLTSolver>(threshold);
                    break;
                case SolverType::CG:
                    solver_ = std::make_unique<LinSolvers::CGSolver>(threshold);
                    break;
                case SolverType::LSCG:
                    solver_ = std::make_unique<LinSolvers::LSCGSolver>(threshold);
                    break;
                case SolverType::BICGSTAB:
                    solver_ = std::make_unique < LinSolvers::BICGSTABSolver>(threshold);
                    break;
                default:
                    std::cerr << "Unkown solver, defaulting to LU Solver" << std::endl;
                    solver_ = std::make_unique < LinSolvers::LUSolver>(threshold);
            }
        }

        inline Eigen::VectorXd solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b, const Eigen::VectorXd& initialGuess) const {
            return solver_->solve(A, b, initialGuess);
        }
    };
}

#endif
