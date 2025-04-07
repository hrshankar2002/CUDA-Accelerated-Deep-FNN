# Compilers
NVCC = nvcc
CXX = g++

# Targets
TARGET_CUDA = test.o
TARGET_CPU = main

# Source files
SRC_CUDA = cuda_mlp_v1.cu
SRC_CPU = cpu_mlp_v1.cpp

# Default paths (users can override these by setting environment variables)
PYTHON_INCLUDE ?= /path/to/python/include
PYTHON_LIB ?= /path/to/python/lib
NUMPY_INCLUDE ?= /path/to/numpy/include

# Include directories
INCLUDES = -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE)

# Library directories
LIBDIRS = -L$(PYTHON_LIB)

# Libraries to link
LIBS = -lpython3.3m

# Compilation flags
CFLAGS_CUDA = $(INCLUDES) $(LIBDIRS) $(LIBS)
CFLAGS_CPU = -std=c++11 $(INCLUDES) $(LIBDIRS) $(LIBS)

# Rules
all: $(TARGET_CUDA) $(TARGET_CPU)

$(TARGET_CUDA): $(SRC_CUDA)
	$(NVCC) -o $(TARGET_CUDA) $(SRC_CUDA) $(CFLAGS_CUDA)

$(TARGET_CPU): $(SRC_CPU)
	$(CXX) -o $(TARGET_CPU) $(SRC_CPU) $(CFLAGS_CPU)

run_cuda: $(TARGET_CUDA)
	export LD_LIBRARY_PATH=$(PYTHON_LIB):$$LD_LIBRARY_PATH; \
	export PYTHONHOME=$(PYTHON_INCLUDE); \
	./$(TARGET_CUDA)

run_cpu: $(TARGET_CPU)
	export LD_LIBRARY_PATH=$(PYTHON_LIB):$$LD_LIBRARY_PATH; \
	export PYTHONHOME=$(PYTHON_INCLUDE); \
	./$(TARGET_CPU)

clean:
	rm -f $(TARGET_CUDA) $(TARGET_CPU)
