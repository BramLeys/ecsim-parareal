CXX = g++
MODE = PERFORMANCE
DEBUG_FLAGS = -Wall -Wextra -std=c++20 -fopenmp -Og
PERFORMANCE_FLAGS = -std=c++20 -fopenmp -O3 -DNDEBUG -ftree-vectorize -march=native
CXXFLAGS = $($(MODE)_FLAGS)
SRCDIR = .
INCDIR = .
TESTDIR = ./tests
EIGEN_DIR = ../../../../../../../../Dev/eigen-3.4.0

# Find all source files
SRCS = $(wildcard $(SRCDIR)/*.cpp)
HDR = $(wildcard $(INCDIR)/*.h)
TSTH = $(wildcard $(TESTDIR)/*.h)

all: test CNConvtest ECSIM1DConvtest ECSIM3DConvtest ECSIM3DSim ECSIMConvtest

# Rule to build the target executable
% : $(SRCDIR)/%.cpp $(HDR) $(TSTH)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -I$(EIGEN_DIR) $< -o $@.exe 

clean:
	rm -f *.exe
