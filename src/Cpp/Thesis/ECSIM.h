#ifndef ECSIM_SOLVERS
#define ECSIM_SOLVERS
#include "debug.h"
#include "common.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/IterativeSolvers>
using namespace Eigen;
namespace TestProblems {

    // 1D-3V
    int SetTransverse(ArrayXd& xp, Array3Xd& vp, Array3Xd& E0, Array3Xd& Bc, ArrayXd& qp, int Nx = 128, int Np = 10000, double L = 2 * EIGEN_PI) {
        double dx = L / Nx; 
        double Vx = pow(dx, 1); // volumes of each of the grid cells (need regular grid)
        double VT = 0.01; // thermic velocity of particles in each direction
        double V0 = 0.2;

        xp = ArrayXd::LinSpaced(Np, 0, L - L / Np);
        vp = (VT * Array3Xd::Random(3, Np));
        E0 = Array3Xd::Ones(3, Nx)/10;//initialization of electric field in each of the grid cells
        //E0 = Array3Xd::Zero(3, Nx);
        Bc = Array3Xd::Zero(3, Nx);
        Bc.row(0) = ArrayXd::Ones(Nx) / 10;

        ArrayXd pm =ArrayXd::LinSpaced(Np, 1, Np);
        pm = 1 - 2 * mod(pm, 2).cast<double>();
        vp.row(1) += pm * V0;

        ArrayXi ix = (xp / dx).floor().cast<int>(); // cell of the particle, first cell is cell 0, first node is 0 last node Nx
        ArrayXd frac1 = 1 - (xp / dx - ix.cast<double>()); // W_{pg}
        ArrayXi ix2 = mod((ix + 1), Nx).cast<int>(); // second cell of influence due to extended local support of first order b-spline

        //watch(ix);
        //watch(ix2);
        //watch(frac1);
        std::vector<Triplet<double>> tripletList(Np * 4);
        #pragma omp parallel for
        for (int ip = 0; ip < Np; ip++) {
            tripletList[ip * 4] = Triplet<double>(ix(ip), ix(ip), pow(frac1(ip), 2));
            tripletList[ip * 4 + 1] = Triplet<double>(ix2(ip), ix(ip), frac1(ip) * (1 - frac1(ip)));
            tripletList[ip * 4 + 2] = Triplet<double>(ix(ip), ix2(ip), frac1(ip) * (1 - frac1(ip)));
            tripletList[ip * 4 + 3] = Triplet<double>(ix2(ip), ix2(ip), pow((1 - frac1(ip)), 2));
        }
        SparseMatrix<double> M(Nx, Nx);
        M.setFromTriplets(tripletList.begin(), tripletList.end());
        //watch(M.toDense());
        VectorXd rhotarget = -VectorXd::Ones(Nx) * Vx;
        SparseLU<SparseMatrix<double>> solver;
        solver.compute(M);
        if (solver.info() != Success) {
            // decomposition failed
            std::cerr << "Decomposition failed" << std::endl;
            return 1;
        }
        auto rhotildeV = solver.solve(rhotarget);
        if (solver.info() != Success) {
            // solving failed
            std::cerr << "Solving failed" << std::endl;
            return 1;
        }
        //watch(rhotildeV);
        qp = rhotildeV.array()(ix) * frac1 + rhotildeV.array()(ix2) * (1 - frac1);
        return 0;
    }

    // 1D-1V
    int SetTwoStream(ArrayXd& xp, ArrayXd& vp, ArrayXd& E0, ArrayXd& Bc, ArrayXd& qp, int Nx = 128, int Np = 10000, double L = 2 * EIGEN_PI) {
        double dx = L / Nx;
        double Vx = pow(dx, 1); // volumes of each of the grid cells (need regular grid)
        double mode = 5;    // mode of perturbation sine
        double VT = 0.02; // thermic velocity of particles in each direction
        double V0 = 0.1;
        double V1 = .1 * V0;

        xp = ArrayXd::LinSpaced(Np, 0, L - L / Np);
        vp = (VT * ArrayXd::Random(Np));
        E0 = ArrayXd::Zero(Nx);//initialization of electric field in each of the grid cells
        Bc = ArrayXd::Ones(Nx) / 10;

        ArrayXd pm = 1 + ArrayXd::LinSpaced(Np, 0, Np - 1);
        pm = 1 - 2 * mod(pm, 2).cast<double>();
        vp += pm * V0;

        vp += V1 * (2 * EIGEN_PI * xp / L * mode).sin();

        ArrayXi ix = (xp / dx).floor().cast<int>(); // cell of the particle, first cell is cell 0, first node is 0 last node Nx
        ArrayXd frac1 = 1 - (xp / dx - ix.cast<double>()); // W_{pg}
        ArrayXi ix2 = mod((ix + 1), Nx).cast<int>(); // second cell of influence due to extended local support of first order b-spline

        std::vector<Triplet<double>> tripletList(Np * 4);
        #pragma omp parallel for shared(tripletList)
        for (int ip = 0; ip < Np; ip++) {
            tripletList[ip * 4] = Triplet<double>(ix(ip), ix(ip), pow(frac1(ip), 2));
            tripletList[ip * 4 + 1] = Triplet<double>(ix2(ip), ix(ip), frac1(ip) * (1 - frac1(ip)));
            tripletList[ip * 4 + 2] = Triplet<double>(ix(ip), ix2(ip), frac1(ip) * (1 - frac1(ip)));
            tripletList[ip * 4 + 3] = Triplet<double>(ix2(ip), ix2(ip), pow((1 - frac1(ip)), 2));
        }
        SparseMatrix<double> M(Nx, Nx);
        M.setFromTriplets(tripletList.begin(), tripletList.end());

        VectorXd rhotarget = -VectorXd::Ones(Nx) * Vx;
        SimplicialLDLT<SparseMatrix<double>> solver;
        solver.compute(M);
        if (solver.info() != Success) {
            // decomposition failed
            std::cerr << "Decomposition failed" << std::endl;
            return 1;
        }
        auto rhotildeV = solver.solve(rhotarget);
        if (solver.info() != Success) {
            // solving failed
            std::cerr << "Solving failed" << std::endl;
            return 1;
        }

        qp = rhotildeV.array()(ix) * frac1 + rhotildeV.array()(ix2) * (1 - frac1);
        return 0;
    }

    // 1D-1V
    int SetConvergence(ArrayXd& xp, ArrayXd& vp, ArrayXd& E0, ArrayXd& Bc, ArrayXd& qp, int Nx = 128, int Np = 10000, double L = 2 * EIGEN_PI) {
        double V0 = 0.1;
        double V1 = 0.1 * V0;
        double mode = 5;
        xp = ArrayXd::LinSpaced(Np, 0, L - L / Np);
        vp = V1 * (xp * 2 * EIGEN_PI / L * mode).sin();
        E0 = (3 * ArrayXd::LinSpaced(Nx, 0, L - L / Nx)).cos();
        Bc = ArrayXd::Zero(Nx);
        qp = -L * ArrayXd::Ones(Np);
        return 0;
    }

    // 1D-3V
    int SetConvergence(ArrayXd& xp, Array3Xd& vp, Array3Xd& E0, Array3Xd& Bc, ArrayXd& qp, int Nx = 128, int Np = 10000, double L = 2 * EIGEN_PI) {
        double dx = L / Nx;
        double Vx = pow(L / Np, 1);
        double V1 = 0.1;
        double mode = 5;
        xp = ArrayXd::LinSpaced(Np, 0, L - L / Np);
        vp.row(0) = V1 * (xp * 2 * EIGEN_PI / L * mode).sin();
        vp.row(1) = V1 * (xp * 2 * EIGEN_PI / L * (mode+1)).sin();
        vp.row(2) = V1 * (xp * 2 * EIGEN_PI / L * (mode + 2)).sin();
        E0.row(0) = (3 * ArrayXd::LinSpaced(Nx, 0, L - L / Nx).cos());
        E0.row(1) = (4 * ArrayXd::LinSpaced(Nx, 0, L - L / Nx).cos());
        E0.row(2) = (2 * ArrayXd::LinSpaced(Nx, 0, L - L / Nx).cos());
        Bc = Array3Xd::Ones(3,Nx);
        qp = -L * ArrayXd::Ones(Np);
        return 0;
    }
}

template <int xdim,int vdim>
class ECSIMBase {
protected:
    double L; // length of position space
    int Np; // number of particles
    int Nx; // number of grid cells
    int Nsub; // number of subcycles for each timestep
    double dx; // length of each grid cell
    double dt; //time step length
    double qom = -1; //charge mass density of particles (q/m)
    double theta = 0.5; //theta of field and particle mover
    const ArrayXd& qp; // charge of each particle
    double Vx; // volume of regular grid
public:
    ECSIMBase(double L, int Np, int Nx, int Nsub, double dt, Eigen::ArrayXd const& qp)
        :L(L), Np(Np), Nx(Nx), Nsub(Nsub), dt(dt), qp(qp)
    {
        dx = L / Nx;
        Vx = pow(dx, xdim);
    }

    inline double Get_dt() { return dt; }
    inline void Set_dt(double timestep) { dt = timestep; }

    inline int Get_Nsub() { return Nsub; }
    inline void Set_Nsub(int subcycles) { Nsub = subcycles; }

    inline int Get_vdim() { return vdim; }
    inline int Get_xdim() { return xdim; }

    inline Eigen::Array3d Energy(const Eigen::Ref<const MatrixXd> Xn) const {
        Eigen::Map < const Array<double, vdim, -1>> vp(Xn.data() + xdim * Np, vdim, Np);
        Eigen::Map < const Array<double, vdim, -1>> E0(Xn.data() + (xdim + vdim) * Np, vdim, Nx);
        Eigen::Map < const Array<double, vdim, -1>> Bc(Xn.data() + xdim * Np + (Np + Nx) * vdim, vdim, Nx);
        Eigen::Array3d energies;
        energies(0) = 0.5 * (qp.transpose() * vp.colwise().squaredNorm()).sum() / qom;
        energies(1) = 0.5 * E0.matrix().squaredNorm() * dx;
        energies(2) = 0.5 * Bc.matrix().squaredNorm() * dx;
        //energies << 0.5 * (qp.transpose() * vp.colwise().squaredNorm()).sum() / qom, 0.5 * E0.matrix().squaredNorm() * dx, 0.5 * Bc.matrix().squaredNorm() * dx;
        return energies;

    }

    inline Eigen::ArrayXXd Divergence(const Eigen::Ref<const MatrixXd> X) const {
        ArrayXXd div = ArrayXXd::Zero(xdim * this->Nx, X.cols());
        for (int i = 0; i < X.cols(); i++) {
            Eigen::Map <const Matrix<double, vdim, -1>> Bc(X.col(i).data() + xdim * this->Np + (this->Np + this->Nx) * vdim, vdim, this->Nx);
            for (int j = 0; j < this->Nx; j++) {
                for (int k = 0; k < xdim; k++) {
                    // centered difference
                    div(j,i) += (Bc(k, (j + 1 + this->Nx) % this->Nx) - Bc(k, (j - 1 + this->Nx) % this->Nx)) / (2*this->dx);
                }
            }
        }
        return div;
    }

    inline Eigen::Array4Xd Error(const Eigen::Ref<const MatrixXd> X, const Eigen::Ref<const MatrixXd> Y) const {

        Eigen::Array4Xd errors = Array4Xd::Zero(4,X.cols());
        for (int i = 0; i < X.cols(); i++) {
            Eigen::Map <const Matrix<double, xdim, -1>> X_xp(X.col(i).data(), xdim, this->Np);
            Eigen::Map <const Matrix<double, vdim, -1>> X_vp(X.col(i).data() + xdim * this->Np, vdim, this->Np);
            Eigen::Map <const Matrix<double, vdim, -1>> X_E0(X.col(i).data() + (xdim + vdim) * this->Np, vdim, this->Nx);
            Eigen::Map <const Matrix<double, vdim, -1>> X_Bc(X.col(i).data() + xdim * this->Np + (this->Np + this->Nx) * vdim, vdim, this->Nx);

            Eigen::Map <const Matrix<double, xdim, -1>> Y_xp(Y.col(i).data(), xdim, this->Np);
            Eigen::Map <const Matrix<double, vdim, -1>> Y_vp(Y.col(i).data() + xdim * this->Np, vdim, this->Np);
            Eigen::Map <const Matrix<double, vdim, -1>> Y_E0(Y.col(i).data() + (xdim + vdim) * this->Np, vdim, this->Nx);
            Eigen::Map <const Matrix<double, vdim, -1>> Y_Bc(Y.col(i).data() + xdim * this->Np + (this->Np + this->Nx) * vdim, vdim, this->Nx);

            errors.col(i) << (X_xp - Y_xp).norm() / X_xp.norm(), (X_vp - Y_vp).norm() / X_vp.norm(), (X_E0 - Y_E0).norm() / X_E0.norm(), (X_Bc - Y_Bc).norm() / X_Bc.norm();
            //PRINT("Total error:", ((X - Y).colwise().norm().norm() / (X.norm())));
            //PRINT("Split errors:", errors);

        }
        return errors;
    }

};

template <int xdim, int vdim>
class ECSIM : public ECSIMBase<xdim, vdim> {
public:
    using ECSIMBase<xdim, vdim>::ECSIMBase;

    // yn is a 1D array and will only contain the state at t1
    inline void Step(const Eigen::Ref<const MatrixXd> xn, double t0, double t1, Eigen::Ref<MatrixXd> yn) const {
        throw std::runtime_error("Only vdim == 1 or vdim == 3 are supported");
    }
    
    // yn is a 2D array and will contain the full simulation on exit
    void Solve(const Eigen::Ref<const MatrixXd> xn, double t0, double t1, Eigen::Ref<MatrixXd> yn) const {
        throw std::runtime_error("Only vdim == 1 or vdim == 3 are supported");
    }
};

template <int xdim>
class ECSIM<xdim,1>: public ECSIMBase<xdim, 1> {
public:
    using ECSIMBase<xdim, 1>::ECSIMBase;

    // yn is a 1D array and will only contain the state at t1
    inline void Step(const Eigen::Ref<const MatrixXd> xn, double t0, double t1, Eigen::Ref<MatrixXd> yn, bool adjust_init = false) const {
        int nb_steps = round(abs(t1 - t0) / this->dt);
        MatrixXd steps(yn.rows(), nb_steps + 1);
        VectorXd x0 = xn;
        // The actual initial position should be half of a timestep before t0 -> adjust if necessary
        if (adjust_init) {
            Eigen::Map <const Array<double, 1, -1>> vp(x0.data() + xdim * this->Np, 1, this->Np);
            x0 << x0.head(xdim*this->Np).array() - vp.row(0).array() * this->dt / 2, x0.tail(x0.rows() - xdim*this->Np);
        }
        Solve(x0, t0, t1, steps);
        yn = steps.col(nb_steps);
        return;
    }
    
    // yn is a 2D array and will contain the full simulation on exit
    void Solve(const Eigen::Ref<const MatrixXd> xn, double t0, double t1, Eigen::Ref<MatrixXd> yn) const {
        yn.col(0) = xn;

        ArrayXd x_view = xn.col(0).head(this->Np);
        ArrayXd vp = xn.col(0).segment(this->Np, this->Np);
        ArrayXd E0 = xn.col(0).segment(2 * this->Np, this->Nx);
        ArrayXd Bc = xn.col(0).tail(this->Nx);

        auto oldE = this->Energy(xn);

        Eigen::ArrayXXi ix(this->Np, this->Nsub);
        Eigen::ArrayXXd frac1(this->Np, this->Nsub);
        Eigen::ArrayXXi ix2(this->Np, this->Nsub);

        std::vector<Triplet<double>> tripletListfrac(2 * this->Np * this->Nsub);
        Eigen::SparseMatrix<double> frac_p(this->Nx, this->Np);

        Eigen::ArrayXd J0(this->Nx);

        Eigen::SparseMatrix<double> M(this->Nx, this->Nx);

        Eigen::SparseMatrix<double> NxIdentity(this->Nx, this->Nx);
        NxIdentity.setIdentity();

        Eigen::VectorXd bKrylov(this->Nx);
        Eigen::VectorXd xKrylov(this->Nx);
        Eigen::SparseMatrix<double> Maxwell(this->Nx, this->Nx);

        SparseLU<SparseMatrix<double>> solver;

        // Eigen doesn't want to cast when this is not done
        int Nx = this->Nx;
        double dx = this->dx;

        for (int step = 0; step < yn.cols() - 1; step++) {
            auto tic = std::chrono::high_resolution_clock::now();
            for (int itsub = 0; itsub < this->Nsub; itsub++) {
                x_view += vp * this->dt / this->Nsub;
                // Position is periodic with period L (parareal can go out of bounds between solves)
                x_view = mod(x_view, this->L);

                ix.col(itsub) = (x_view / dx).floor().cast<int>(); // cell of the particle, first cell is cell 0, first node is 0 last node this->Nx
                frac1.col(itsub) = 1 - (x_view / this->dx - ix.col(itsub).cast<double>()); // W_{pg}
                ix2.col(itsub) = mod((ix.col(itsub) + 1), Nx).cast<int>(); // second cell of influence due to extended local support of first order b-spline

                for (int ip = 0; ip < this->Np; ip++) {
                    tripletListfrac[2 * (itsub * this->Np + ip)] = Triplet<double>(ix(ip, itsub), ip, frac1(ip, itsub) / this->Nsub);
                    tripletListfrac[2 * (itsub * this->Np + ip) + 1] = Triplet<double>(ix2(ip, itsub), ip, (1 - frac1(ip, itsub)) / this->Nsub);
                }
            }
            frac_p.setFromTriplets(tripletListfrac.begin(), tripletListfrac.end());
            J0 = frac_p * (this->qp * vp / this->Vx).matrix();
            M = (frac_p * this->qp.matrix().asDiagonal()) * frac_p.transpose();

            bKrylov = E0 - J0 * this->dt * this->theta;
            Maxwell = NxIdentity + this->qom * this->dt * this->dt * this->theta / 2 / this->dx * M;

            solver.compute(Maxwell);
            if (solver.info() != Success) {
                // decomposition failed
                std::cerr << "Decomposition of Maxwell matrix failed" << std::endl;
                return;
            }
            xKrylov = solver.solve(bKrylov);
            if (solver.info() != Success) {
                // solving failed
                std::cerr << "Solving failed" << std::endl;
                return;
            }

            E0 = (xKrylov.array() - E0 * (1 - this->theta)) / this->theta;
            for (int itsub = 0; itsub < this->Nsub; itsub++) {
                vp += this->dt / this->Nsub * (xKrylov.array()(ix.col(itsub)) * frac1.col(itsub) + xKrylov.array()(ix2.col(itsub)) * (1 - frac1.col(itsub))) * this->qom;
            }

            yn.col(step + 1) << x_view, vp, E0, Bc;
            auto toc = std::chrono::high_resolution_clock::now();
            double diff = abs((this->Energy(yn.col(step + 1)) - oldE).sum()) / abs(oldE.sum());
            //PRINT("Finished timestep", step, " in ", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms with ", this->Nsub,"subcycles and energy conservation equal to ",diff);
        }
    }
};

template <int xdim>
class ECSIM<xdim, 3> :public ECSIMBase<xdim, 3> {
public:
    using ECSIMBase<xdim, 3>::ECSIMBase;

    //yn is a 1D array and will only contain the state at t1
    inline void Step(const Eigen::Ref<const MatrixXd> xn, double t0, double t1, Eigen::Ref<MatrixXd> yn, bool adjust_init = false) const {
        int nb_steps = round(abs(t1 - t0) / this->dt);
        MatrixXd steps(yn.rows(), nb_steps + 1);
        VectorXd x0 = xn;
        // The actual initial position should be half of a timestep before t0 -> adjust if necessary
        if (adjust_init) {
            Eigen::Map <const Array<double, 3, -1>> vp(x0.data() + xdim * this->Np, 3, this->Np);
            x0 << x0.head(this->Np).array() - vp.row(0).array() * this->dt / 2, x0.tail(x0.rows() - this->Np);
        }
        Solve(x0, t0, t1, steps);
        yn = steps.col(nb_steps);
        return;
    }


    //yn is a 2D array and will contain the full simulation on exit
    void Solve(const Eigen::Ref<const MatrixXd> xn, double t0, double t1, Eigen::Ref<MatrixXd> yn) const {
        yn.col(0) = xn;

        ArrayXd x_view = yn.col(0).head(this->Np);
        Eigen::Map <Array<double, 3, -1>> vp(yn.col(0).data() + xdim * this->Np, 3, this->Np);
        Eigen::Map <Array<double, 3, -1>> E0(yn.col(0).data() + (xdim + 3) * this->Np, 3, this->Nx);
        Eigen::Map <Array<double, 3, -1>> Bc(yn.col(0).data() + xdim * this->Np + (this->Np + this->Nx) * 3, 3, this->Nx);

        Eigen::ArrayXXi ix(this->Np, this->Nsub);
        Eigen::ArrayXXd frac1(this->Np, this->Nsub);
        Eigen::ArrayXXi ix2(this->Np, this->Nsub);

        Eigen::Array3Xd B(3, this->Nx);
        std::vector<Eigen::Matrix3d> alphap(this->Np * this->Nsub); // column major
        //Eigen::Array3Xd vphat(3,this->Np);
        Eigen::Array3Xd J0(Bc.rows(), Bc.cols());

        std::vector<std::vector<Triplet<double>>> tripletListM(9,std::vector<Triplet<double>>(this->Np * 4 * this->Nsub));
        std::vector<Eigen::SparseMatrix<double>> Ms(9); // column major storage

        Eigen::SparseMatrix<double> AmpereX(this->Nx, this->Nx);
        Eigen::SparseMatrix<double> AmpereY(this->Nx, this->Nx);
        Eigen::SparseMatrix<double> AmpereZ(this->Nx, this->Nx);

        Eigen::SparseMatrix<double> Derv(this->Nx, this->Nx);
        Eigen::SparseMatrix<double> Derc(this->Nx, this->Nx);
        std::vector<Triplet<double>> tripletListDerc(this->Nx * 2);
        std::vector<Triplet<double>> tripletListDerv(this->Nx * 2);

        Eigen::SparseMatrix<double> NxIdentity(this->Nx, this->Nx);
        NxIdentity.setIdentity();
        Eigen::SparseMatrix<double> NxZeros(this->Nx, this->Nx);
        NxZeros.setZero();

        Eigen::VectorXd bKrylov(5 * this->Nx);
        Eigen::VectorXd xKrylov = VectorXd::Zero(5 * this->Nx);

        Eigen::SparseMatrix<double> Maxwell(5 * this->Nx, 5 * this->Nx);

        SparseLU<SparseMatrix<double>> solver;
        //GMRES<SparseMatrix<double>, IncompleteLUT<double>> solver;

        std::vector<Triplet<double>> tripletListMaxwell;

        double beta = this->qom * this->dt / 2 / this->Nsub;

        int Nx = this->Nx;
        double dx = this->dx;

        auto oldE = this->Energy(xn);

        ArrayXXd cycle_steps(x_view.rows(),this->Nsub);
        std::vector<Array3Xd> Bp(this->Nsub, Array3Xd(3, this->Np));
        std::vector<Array3Xd> vphat(this->Nsub, Array3Xd(3, this->Np));

        for (size_t step = 0; step < yn.cols() - 1; step++) {
            auto tic = std::chrono::high_resolution_clock::now();
            B.rightCols(B.cols() - 1) = 0.5 * (Bc.rightCols(B.cols() - 1) + Bc.leftCols(B.cols() - 1));
            B.col(0) = 0.5 * (Bc.col(B.cols() - 1) + Bc.col(0));

            //watch(B);

            J0.setZero();
            #pragma omp parallel for num_threads(this->Nsub)
            for (int itsub = 0; itsub < this->Nsub; itsub++){
                cycle_steps.col(itsub) = x_view + vp.row(0).transpose() * (itsub + 1) * this->dt / this->Nsub;
                //x_view += vp.row(0) * itsub * this->dt / this->Nsub;
                // Position is periodic with period L (parareal can go out of bounds between solves)
                cycle_steps.col(itsub) = mod(cycle_steps.col(itsub), this->L);
                //cycle_steps.col(itsub) = x_view;

                ix.col(itsub) = (cycle_steps.col(itsub) / dx).floor().cast<int>(); // cell of the particle, first cell is cell 0, first node is 0 last node this->Nx
                frac1.col(itsub) = 1 - (cycle_steps.col(itsub) / this->dx - ix.col(itsub).cast<double>()); // W_{pg}
                ix2.col(itsub) = mod((ix.col(itsub) + 1), Nx).cast<int>(); // second cell of influence due to extended local support of first order b-spline

                for (int ip = 0; ip < this->Np; ip++) {
                    Bp[itsub].col(ip) = B.col(ix(ip, itsub)) * frac1(ip, itsub) + B.col(ix2(ip, itsub)) * (1 - frac1(ip, itsub));
                    double sx = Bp[itsub](0, ip) * beta;
                    double sy = Bp[itsub](1, ip) * beta;
                    double sz = Bp[itsub](2, ip) * beta;

                    alphap[itsub * this->Np + ip].col(0) << 1 + sx * sx, -sz + sx * sy, sx* sz + sy;
                    alphap[itsub * this->Np + ip].col(1) << sz + sx * sy, 1 + sy * sy, -sx + sy * sz;
                    alphap[itsub * this->Np + ip].col(2) << sx * sz - sy, sx + sy * sz, 1 + sz * sz;
                    alphap[itsub * this->Np + ip].array() /= 1 + (Bp[itsub].col(ip) * (beta)).pow(2).sum();

                    vphat[itsub].col(ip) = alphap[itsub * this->Np + ip] * vp.col(ip).matrix();

                }
                for (int j = 0; j < 3; j++) {
                    for (int i = 0; i < 3; i++) {
                        for (int ip = 0; ip < this->Np; ip++) {
                            tripletListM[3 * j + i][itsub * 4 * this->Np + ip * 4]     = Triplet<double>(ix(ip, itsub), ix(ip, itsub), pow(frac1(ip, itsub), 2) * this->qp(ip) * alphap[itsub * this->Np + ip](i, j) / this->Nsub);
                            tripletListM[3 * j + i][itsub * 4 * this->Np + ip * 4 + 1] = Triplet<double>(ix2(ip, itsub), ix(ip, itsub), frac1(ip,itsub) * (1 - frac1(ip, itsub)) * this->qp(ip) * alphap[itsub * this->Np + ip](i, j) / this->Nsub);
                            tripletListM[3 * j + i][itsub * 4 * this->Np + ip * 4 + 2] = Triplet<double>(ix(ip, itsub), ix2(ip, itsub), frac1(ip,itsub) * (1 - frac1(ip, itsub)) * this->qp(ip) * alphap[itsub * this->Np + ip](i, j) / this->Nsub);
                            tripletListM[3 * j + i][itsub * 4 * this->Np + ip * 4 + 3] = Triplet<double>(ix2(ip, itsub), ix2(ip, itsub), pow((1 - frac1(ip, itsub)), 2) * this->qp(ip) * alphap[itsub * this->Np + ip](i, j) / this->Nsub);
                        }
                    }
                }
            }
            for (int itsub = 0; itsub < this->Nsub; itsub++) {
                for (int ip = 0; ip < this->Np; ip++) {
                    J0.col(ix(ip, itsub)) += frac1(ip, itsub) * this->qp(ip) * vphat[itsub].col(ip);
                    J0.col(ix2(ip, itsub)) += (1 - frac1(ip, itsub)) * this->qp(ip) * vphat[itsub].col(ip);
                }
            }

            J0 /= this->Vx * this->Nsub;
            // Setting Ms in columnmajor ordering -> [M0_0, M1_0, M2_0, M0_1, M1_1, M2_1, ...]
            for (int j = 0; j < 3; j++) {
                for (int i = 0; i < 3; i++) {
                    Ms[3 * j + i].resize(this->Nx, this->Nx);
                    Ms[3 * j + i].setFromTriplets(tripletListM[3 * j + i].begin(), tripletListM[3 * j + i].end());
                }
            }

            AmpereX = NxIdentity + this->qom * this->dt * this->dt * this->theta / 2 * Ms[0] / this->dx;
            AmpereY = NxIdentity + this->qom * this->dt * this->dt * this->theta / 2 * Ms[4] / this->dx;
            AmpereZ = NxIdentity + this->qom * this->dt * this->dt * this->theta / 2 * Ms[8] / this->dx;

            for (int i = 0; i < this->Nx; i++) {
                tripletListDerv[2 * i] = Triplet<double>(i, i, 1 / this->dx* this->dt * this->theta);
                tripletListDerv[2 * i + 1] = Triplet<double>(i, (i - 1 + this->Nx) % this->Nx, -1 / this->dx * this->dt * this->theta);

                tripletListDerc[2 * i] = Triplet<double>(i, i, -1 / this->dx * this->dt * this->theta);
                tripletListDerc[2 * i + 1] = Triplet<double>(i, (i + 1) % this->Nx,  1 / this->dx * this->dt * this->theta);
            }
            Derv.setFromTriplets(tripletListDerv.begin(), tripletListDerv.end());
            Derc.setFromTriplets(tripletListDerc.begin(), tripletListDerc.end());

            bKrylov << (E0.row(0) - J0.row(0) * this->dt * this->theta).transpose(), (E0.row(1) - J0.row(1) * this->dt * this->theta).transpose(), (E0.row(2) - J0.row(2) * this->dt * this->theta).transpose(), Bc.row(1).transpose(), Bc.row(2).transpose();
            tripletListMaxwell.resize(AmpereX.nonZeros() + AmpereY.nonZeros() + AmpereZ.nonZeros() + Ms[1].nonZeros() + Ms[2].nonZeros() + Ms[3].nonZeros() +
                Ms[5].nonZeros() + Ms[6].nonZeros() + Ms[7].nonZeros() + 2 * Derv.nonZeros() + 2 * Derc.nonZeros() + 2 * this->Nx);

            int index = 0;
            // First Rowblock
            for (int k = 0; k < AmpereX.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(AmpereX, k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row(), it.col(), it.value());
                }
            for (int k = 0; k < Ms[3].outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Ms[3], k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row(), it.col() + this->Nx, this->qom * this->dt * this->dt * this->theta / 2 / this->dx * it.value());
                }

            for (int k = 0; k < Ms[6].outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Ms[6], k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row(), it.col() + 2 * this->Nx, this->qom * this->dt * this->dt * this->theta / 2 / this->dx * it.value());
                }

            // Second Rowblock
            for (int k = 0; k < Ms[1].outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Ms[1], k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 1 * this->Nx, it.col() + 0 * this->Nx, this->qom * this->dt * this->dt * this->theta / 2 / this->dx * it.value());
                }
            
            for (int k = 0; k < AmpereY.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(AmpereY, k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 1 * this->Nx, it.col() + 1 * this->Nx, it.value());
                }
            for (int k = 0; k < Ms[7].outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Ms[7], k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 1 * this->Nx, it.col() + 2 * this->Nx, this->qom * this->dt * this->dt * this->theta / 2 / this->dx * it.value());
                }
            for (int k = 0; k < Derv.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Derv, k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 1 * this->Nx, it.col() + 4 * this->Nx, it.value());
                }

            // Third Rowblock
            for (int k = 0; k < Ms[2].outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Ms[2], k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 2 * this->Nx, it.col() + 0 * this->Nx, this->qom * this->dt * this->dt * this->theta / 2 / this->dx * it.value());
                }
            for (int k = 0; k < Ms[5].outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Ms[5], k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 2 * this->Nx, it.col() + 1 * this->Nx, this->qom * this->dt * this->dt * this->theta / 2 / this->dx * it.value());
                }
            for (int k = 0; k < AmpereZ.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(AmpereZ, k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 2 * this->Nx, it.col() + 2 * this->Nx, it.value());
                }
            for (int k = 0; k < Derv.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Derv, k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 2 * this->Nx, it.col() + 3 * this->Nx, -it.value());
                }

            // Fourth Rowblock
            for (int k = 0; k < Derc.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Derc, k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 3 * this->Nx, it.col() + 2 * this->Nx, -it.value());
                }
            for (int k = 0; k < this->Nx; ++k) {
                tripletListMaxwell[index++] = Triplet<double>(k + 3 * this->Nx, k + 3 * this->Nx, 1);
            }
            
            // Fifth Row
            for (int k = 0; k < Derc.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(Derc, k); it; ++it)
                {
                    tripletListMaxwell[index++] = Triplet<double>(it.row() + 4 * this->Nx, it.col() + 1 * this->Nx, it.value());
                }
            for (int k = 0; k < this->Nx; ++k) {
                tripletListMaxwell[index++] = Triplet<double>(k + 4 * this->Nx, k + 4 * this->Nx, 1);
            }
            Maxwell.setFromTriplets(tripletListMaxwell.begin(), tripletListMaxwell.end());

            //auto x0 = MatrixXd(1,5 * Nx);
            //x0 << E0.row(0), E0.row(1), E0.row(2), Bc.row(1), Bc.row(2);
            solver.compute(Maxwell);
            xKrylov = solver.solve(bKrylov);


            E0 = ((Map < Array<double,3,Dynamic,Eigen::RowMajor>>(xKrylov.data(), 3, this->Nx)) - E0 * (1 - this->theta)) / this->theta;
            Bc.row(1) = (xKrylov(seqN(3 * this->Nx, this->Nx)).array() - Bc.row(1).transpose() * (1 - this->theta)) / this->theta;
            Bc.row(2) = (xKrylov(seqN(4 * this->Nx, this->Nx)).array() - Bc.row(2).transpose() * (1 - this->theta)) / this->theta;

            x_view += vp.row(0) * this->dt;
            //x_view = mod(x_view, this->L);
            //B.rightCols(B.cols() - 1) = 0.5 * (Bc.rightCols(B.cols() - 1) + Bc.leftCols(B.cols() - 1));
            //B.col(0) = 0.5 * (Bc.col(B.cols() - 1) + Bc.col(0));
            
            for (int itsub = 0; itsub < this->Nsub; itsub++) {
                for (int ip = 0; ip < this->Np; ip++) {
                    //Bp[itsub].col(ip) = B.col(ix(ip, itsub)) * frac1(ip, itsub) + B.col(ix2(ip, itsub)) * (1 - frac1(ip, itsub));
                    //double sx = Bp[itsub](0, ip) * beta;
                    //double sy = Bp[itsub](1, ip) * beta;
                    //double sz = Bp[itsub](2, ip) * beta;

                    //alphap[itsub * this->Np + ip].col(0) << 1 + sx * sx, -sz + sx * sy, sx* sz + sy;
                    //alphap[itsub * this->Np + ip].col(1) << sz + sx * sy, 1 + sy * sy, -sx + sy * sz;
                    //alphap[itsub * this->Np + ip].col(2) << sx * sz - sy, sx + sy * sz, 1 + sz * sz;
                    //alphap[itsub * this->Np + ip].array() /= 1 + (Bp[itsub].col(ip) * (beta)).pow(2).sum();
                    vp.col(ip) = (2 * alphap[itsub * this->Np + ip] - Matrix3d::Identity()) * vp.col(ip).matrix() + alphap[itsub * this->Np + ip] * this->qom * this->dt / this->Nsub * (Map < Array<double, 3, Dynamic, Eigen::RowMajor>>(xKrylov.data(), 3, this->Nx).col(ix(ip, itsub)) * frac1(ip, itsub) + Map < Array<double, 3, Dynamic, Eigen::RowMajor>>(xKrylov.data(), 3, this->Nx).col(ix2(ip, itsub)) * (1 - frac1(ip, itsub))).matrix();
                }
            }


            yn.col(step + 1) << x_view, Map<const ArrayXd>(vp.data(), vp.size()), Map<const ArrayXd>(E0.data(), E0.size()), Map<const ArrayXd>(Bc.data(), Bc.size());
            auto toc = std::chrono::high_resolution_clock::now();
            double diff = abs((this->Energy(yn.col(step + 1)) - oldE).sum()) / abs(oldE.sum());
            //PRINT("Finished timestep", step, " in ", std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count(), "ms using ",this->Nsub, "subcycles and with relative energy conservation up to = ", diff);

        }
    }

};


#endif