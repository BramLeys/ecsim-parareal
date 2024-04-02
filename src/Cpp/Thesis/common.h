#ifndef COMMON_FUNC
#define COMMON_FUNC
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>

const std::string filepath = "results/";

template<typename T>
struct ModuloOp {
    const T modulus;
    ModuloOp(const T& modulus) : modulus(modulus) {}
    T operator()(const T& x) const {
        if constexpr (std::is_integral_v<T>) {
            auto result = x % modulus;
            return result + (result < 0) * modulus;
        }
        else {
            auto result = std::fmod(x, modulus);
            return result + (result < 0)* modulus;
        }
    }
};

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
mod(const Eigen::MatrixBase<Derived>& inputMatrix, const typename Derived::Scalar& modulus) {
	return (inputMatrix.array() % modulus).matrix();
}

template<typename Derived>
Eigen::Array<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
mod(const Eigen::ArrayBase<Derived>& inputArray, const typename Derived::Scalar& modulus) {
    return inputArray.unaryExpr(ModuloOp<typename Derived::Scalar>(modulus));
}

//https://libit.sourceforge.net/math_8c-source.html
#define erfinv_a3 -0.140543331
#define erfinv_a2 0.914624893
#define erfinv_a1 -1.645349621
#define erfinv_a0 0.886226899

#define erfinv_b4 0.012229801
#define erfinv_b3 -0.329097515
#define erfinv_b2 1.442710462
#define erfinv_b1 -2.118377725
#define erfinv_b0 1

#define erfinv_c3 1.641345311
#define erfinv_c2 3.429567803
#define erfinv_c1 -1.62490649
#define erfinv_c0 -1.970840454

#define erfinv_d2 1.637067800
#define erfinv_d1 3.543889200
#define erfinv_d0 1

#ifdef WIN32
double erfinv(double x)
{
    it_fprintf(stderr, "undefined function erf()\n");
    return (NAN);
}
#else
double erfinv(double x)
{
    double x2, r, y;
    int  sign_x;

    if (x < -1 || x > 1)
        return NAN;

    if (x == 0)
        return 0;

    if (x > 0)
        sign_x = 1;
    else {
        sign_x = -1;
        x = -x;
    }

    if (x <= 0.7) {

        x2 = x * x;
        r =
            x * (((erfinv_a3 * x2 + erfinv_a2) * x2 + erfinv_a1) * x2 + erfinv_a0);
        r /= (((erfinv_b4 * x2 + erfinv_b3) * x2 + erfinv_b2) * x2 +
            erfinv_b1) * x2 + erfinv_b0;
    }
    else {
        y = sqrt(-log((1 - x) / 2));
        r = (((erfinv_c3 * y + erfinv_c2) * y + erfinv_c1) * y + erfinv_c0);
        r /= ((erfinv_d2 * y + erfinv_d1) * y + erfinv_d0);
    }

    r = r * sign_x;
    x = x * sign_x;

    r -= (erf(r) - x) / (2 / sqrt(EIGEN_PI) * exp(-r * r));
    r -= (erf(r) - x) / (2 / sqrt(EIGEN_PI) * exp(-r * r));

    return r;
}
#endif

#undef erfinv_a3
#undef erfinv_a2
#undef erfinv_a1
#undef erfinv_a0

#undef erfinv_b4
#undef erfinv_b3
#undef erfinv_b2
#undef erfinv_b1
#undef erfinv_b0

#undef erfinv_c3
#undef erfinv_c2
#undef erfinv_c1
#undef erfinv_c0

#undef erfinv_d2
#undef erfinv_d1
#undef erfinv_d0

void save(std::string filename, const Eigen::MatrixXd& x) {
	std::ofstream file(filepath + filename);
	if (file.is_open()) {
        file << std::setprecision(std::numeric_limits<double>::max_digits10);
		file << x;
		file.close();
	}
}

void restore(std::string filename, Eigen::MatrixXd& x) {
	std::ifstream file(filepath + filename);
	if (!file) {
		std::cerr << "Error opening file for reading!" << std::endl;
		return; // Exit with error
	}
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            file >> x(i, j);
        }
    }
    // Close the file
    file.close();
}

#define watch(x) std::cout<<std::setprecision (15)<<"Norm-1 of "<< (#x)<< ": "<< x.abs().sum()<< "   Norm-2 of "<< #x<< ": "<< x.matrix().norm()<< "     Norm-Inf of "<< #x<< ": "<< x.abs().maxCoeff()<<std::endl

#endif
