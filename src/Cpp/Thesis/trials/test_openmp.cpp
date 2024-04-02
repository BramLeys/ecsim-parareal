#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "Timer.hpp"

//! ================================================================= !//

void operation(std::vector<double> &a, std::vector<double> &b, std::vector<double> &c, int N)
{
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii++)
    {
        c[ii] = sin(a[ii]) + cos(b[ii])*sin(b[ii]);
    }
}

void operation_gpu(double* x, double* y, double* z, int N)
{
    #pragma omp target teams distribute parallel for map(to: x[0:N], y[0:N]) map(from: z[0:N])
    for (int ii = 0; ii < N; ii++)
    {
        z[ii] = sin(x[ii]) + cos(y[ii])*sin(y[ii]);
    }
}

void axpby(double a, std::vector<double> &x, double b, std::vector<double> &y, std::vector<double> &z, int N) 
{   
    double tb = omp_get_wtime();
    
    #pragma omp parallel for
    for (int ii = 0; ii < N; ii++) 
    {
        z[ii] = a*x[ii] + b*y[ii];
    }
    
    double te = omp_get_wtime();
    double t = te - tb;
    std::cout << "Time of kernel (CPU):" << t << std::endl;
}

void axpby_gpu(double a, double* x, double b, double* y, double* z, int N) 
{
    #pragma omp target teams distribute parallel for map(to: a, b, x[0:N], y[0:N]) map(from: z[0:N])
    for (int ii = 0; ii < N; ii++) 
    {
        z[ii] = a * x[ii] + b * y[ii];
    }
#pragma omp target 
    {
        if (omp_is_initial_device()) 
        {
            printf("Running on host (CPU)\n");
        } 
        else 
        {
            printf("Running on device (GPU)\n");
        }

    }
}

//! ================================================================= !//

int main(int argc, char** argv)
{

    //? Check OpenMP version
    std::unordered_map<unsigned, std::string> map{  {199810,"1.0"},
                                                    {200203,"2.0"},
                                                    {200505,"2.5"},
                                                    {200805,"3.0"},
                                                    {201107,"3.1"},
                                                    {201307,"4.0"},
                                                    {201511,"4.5"},
                                                    {201811,"5.0"},
                                                    {202011,"5.1"},
                                                    {202111,"5.2"}};
    
    std::cout << "OpenMP version:" << map.at(_OPENMP) << ".\n";

    int on_host;
    int num_devices = omp_get_num_devices();
    printf("Number of available devices %d\n", num_devices);

    for(int ii = 0; ii<num_devices; ii++)
    {
        std::cout << "GPU #:" << num_devices << std::endl;
    }

    int N = atoi(argv[1]);      //* Number of grid points

    std::vector<double> x(N), y(N), z(N);
    
    size_t N_size = N * sizeof(double);
    double *X = (double*)malloc(N_size);
    double *Y = (double*)malloc(N_size);
    double *Z = (double*)malloc(N_size);

    //* Initialise 2 vectors
    for (int ii = 0; ii < N; ii++)
    {
        x[ii] = 1.0*ii;
        y[ii] = 1.0*ii;
        X[ii] = 1.0*ii;
        Y[ii] = 1.0*ii;
    }

    LeXInt::timer time_1, time_2;
    
    // time_1.start();
    // for (int nn = 0; nn < 1; nn++)
    // {
    //     axpby(1.0, x, 1.0, y, z, N);
    //     operation(x, y, z, N);
    // }
    
    // time_1.stop();
    
    time_2.start();
    for (int nn = 0; nn < 1; nn++)
    {
        axpby_gpu(1.0, X, 1.0, Y, Z, N);
        // operation_gpu(X, Y, Z, N);
    }
    time_2.stop();



    // operation(x, y, z, N);

    // std::cout << "CPU code (s): " << time_1.total() << std::endl;
    // std::cout << "GPU code (s): " << time_2.total() << std::endl;

    free(X), free(Y), free(Z);

    return 0;
}