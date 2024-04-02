CXX = g++
MODE = PERFORMANCE
DEBUG_FLAGS = -Wall -Wextra -std=c++20 -fopenmp -Og
PERFORMANCE_FLAGS = -std=c++20 -fopenmp -O3 -DNDEBUG -ftree-vectorize -march=native
CXXFLAGS = $($(MODE)_FLAGS)
SRCDIR = .
BINDIR = bin
INCDIR = .
EIGEN_DIR = ../../../../../../../../Dev/eigen-3.4.0

# Find all source files
SRCS = $(wildcard $(SRCDIR)/*.cpp)
HDR = $(wildcard *.h)
TSTH = $(wildcard tests/*.h)

# Derive object file names from source file names
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(BINDIR)/%.o,$(SRCS))

# Define the target executable
TARGET = test

# Main target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Rule to compile source files into object files
$(BINDIR)/%.o: $(SRCDIR)/%.cpp $(HDR) $(TSTH)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -I$(EIGEN_DIR) -c $< -o $@

# PHONY rule to clean the project
clean:
	rm -f $(BINDIR)/*.o $(TARGET)
