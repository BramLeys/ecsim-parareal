# Master thesis: Parallelisation in time of particle-in-cell plasma simulations using the parareal algorithm

Master thesis of Bram Leys written during the Master in Mathematical Engineering

## Table of Contents
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Sources](#sources)
- [License](#license)
- [Contact](#contact)

## Installation
All results were calculated using the C++ code. There is a makefile which will compile all testcases. The code is dependent on the Eigen library. As such the path to the installation directory should be specified in the makefile.

```bash
git clone https://github.com/BramLeys/ecsim-parareal.git
cd ecsim-parareal/src/Cpp/Thesis/
# change the EIGEN_DIR variable to point to your installation directory of Eigen
nano makefile
```
## Usage
The make file will build and compile everything. Each .cpp file in the src/Cpp/Thesis/ directory is a testcase which can be compiled using "make test". The compiled program can then be run using "./test.exe". Some test cases also have different possible command line arguments. The different arguments can be seen by running "./test.exe -h". To visualize the results you may then use the python scripts in the /src/Cpp/Thesis/results/ directory.

```bash
make testCoarseStep
./testCoarseStep.exe -h
./testCoarseStep.exe -f 1e-5 -nx 512 -n 12 -r 5
```

## Dependencies
### C++
  The project depends on the Eigen library ([https://eigen.tuxfamily.org](https://eigen.tuxfamily.org)), which is licensed under MPL2, and is written for C++20 with OpenMP support.
### Python
  The external libraries used are Numpy ([https://numpy.org](https://numpy.org)) for linear algebra, Scipy ([https://scipy.org](https://scipy.org)) for sparse linear algebra and Matplotlib ([https://matplotlib.org/](https://matplotlib.org/)).

## Sources 
  - J.-L. Lions, Y. Maday, and G. Turinici. Résolution d’EDP par un schéma en
temps «pararéel ». Comptes Rendus de l’Académie des Sciences - Series I -
Mathematics, 332(7):661–668, Apr. 2001.
  - G. Lapenta. Exactly energy conserving semi-implicit particle in cell formulation.
Journal of Computational Physics, 334:349–366, Apr. 2017.
  - G. Lapenta. Advances in the Implementation of the Exactly Energy Conserving
Semi-Implicit (ECsim) Particle-in-Cell Method, volume 5. MDPI AG, Jan. 2023.
  - H. A. van der Vorst. Bi-CGSTAB: A Fast and Smoothly Converging Variant
of Bi-CG for the Solution of Nonsymmetric Linear Systems. SIAM Journal
on Scientific and Statistical Computing, 13(2):631–644, Mar. 1992. Publisher:
Society for Industrial and Applied Mathematics.
  - Youcef Saad, Martin H. Schultz. GMRES: A Generalized Minimal Residual
Algorithm for Solving Nonsymmetric Linear Systems | SIAM Journal on Scientific
Computing, 1986.

## License
This software is licensed under MPL2.

## Contact
  Email: bram.leys@student.kuleuven.be
