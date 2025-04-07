#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <cuda_runtime.h>
#include <numpy/arrayobject.h>


struct Weights {
    float* matrix;
    int ndims;
    int *shape;
    long int size;
};

struct Inputs {
    float* matrix;
    int ndims;
    int *shape;
    long int size;
};

// Numpy Helper Functions:

// 1. Function to read NumPy file:
PyObject* read_numpy_file(const char* file_path) {
    PyObject* numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module == nullptr) {
        PyErr_Print();
        return nullptr;
    }

    PyObject* numpy_function = PyObject_GetAttrString(numpy_module, "load");
    if (numpy_function == nullptr) {
        PyErr_Print();
        Py_DECREF(numpy_module);
        return nullptr;
    }

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(file_path));

    PyObject* result = PyObject_CallObject(numpy_function, args);
    if (result == nullptr) {
        PyErr_Print();
        Py_DECREF(numpy_module);
        Py_DECREF(numpy_function);
        Py_DECREF(args);
        return nullptr;
    }

    Py_DECREF(numpy_module);
    Py_DECREF(numpy_function);
    Py_DECREF(args);
    return result;
}

// 2. Function to read weights and print array:
PyArrayObject* read_weights_from_numpy(const char* file_path, int print) {
    PyObject* numpy_array = read_numpy_file(file_path);
    if (numpy_array == nullptr) {
        return nullptr;
    }

    if (print == 1){
        PyObject* repr = PyObject_Repr(numpy_array);
        const char* str = PyUnicode_AsUTF8(repr);
        printf("%s\n", str);
        Py_DECREF(repr);
    }

    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(numpy_array);
    return array;
}

// 3. Function to return dimension number:
int get_numpy_ndims(PyArrayObject* array){
    int ndim = PyArray_NDIM(array);
    return ndim;
}

// 4. Function to return size:
long int get_numpy_size(PyArrayObject* array){
    npy_intp total_size_intp = PyArray_SIZE(array);
    long int total_size = static_cast<long int>(total_size_intp);
    return total_size;
}

// 5. Convert NumPy array to float and print:
float* convert_PyArrayObject_to_float(PyArrayObject* array, int print, int *shape, int ndim) {
    if (PyArray_TYPE(array) != NPY_FLOAT32) {
        printf("Input numpy array is not of type float.\n");
    }

    float* matrix = static_cast<float*>(PyArray_DATA(array));

    if (print == 1){
        if (ndim == 2){
            for (int i = 0; i < shape[0]; ++i) {
                if (i ==0){
                    for (int j = 0; j < shape[1]; ++j) {
                        if(j<3 || j>shape[1]-3){
                            printf("%d index %.5f ", j+(i*shape[1]), matrix[j+(i*shape[1])]);
                        }
                    }
                    printf("\n");
                }
            }
        }
        else if (ndim == 1){
            for (int i = 0; i < shape[0]; ++i) {
                if(i<5 || i>shape[0]-5){
                    printf("%d index %.5f ", i, matrix[i]);
                }
            }
            printf("\n");
        }
    }

    return matrix;
}

// 6. Move weights to CUDA memory:
float* move_weight_to_cuda(float* weights ,long int total_size){
    float* d_data;
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&d_data, total_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaMemcpy(d_data, weights, total_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    return d_data;
}

// 7. Function to get shape and allocate in memory:
// Note: Here Weights& weights(calling by reference), Weights* weights(calling by pointer) is used
// to avoid dereferencing in accesing objects (. (eg: weights.shape) over -> (eg: weights->shape))
// & no need for nullptr check.
void get_numpy_shape(PyArrayObject* array, Weights& weights, int ndim){
    npy_intp* shape = PyArray_DIMS(array);
    weights.shape = (int *)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        weights.shape[i] = static_cast<int>(shape[i]);
    }
}

// ********************************
// 8. Main function to read weights:
Weights read_weights(const char* file_path, int print){
    Weights weight;
    PyArrayObject* array = read_weights_from_numpy(file_path, print);
    if (array == nullptr) {
        weight.matrix = nullptr;
        return weight;
    }

    int ndims = get_numpy_ndims(array);
    get_numpy_shape(array, weight, ndims);

    long int size = get_numpy_size(array);
    float* matrix = convert_PyArrayObject_to_float(array, print, weight.shape, ndims);

    float* cuda_weights = move_weight_to_cuda(matrix, size);
    Py_DECREF(array);

    weight.ndims = ndims;
    weight.size = size;
    weight.matrix = cuda_weights;

    return weight;
}

// 9. Function to get shape and allocate in memory:
void get_numpy_shape(PyArrayObject* array, Inputs& weights, int ndim){
    npy_intp* shape = PyArray_DIMS(array);
    weights.shape = (int *)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        weights.shape[i] = static_cast<int>(shape[i]);
    }
}

// 10. Main function to read image:
Inputs read_image(const char* file_path, int print){
    Inputs input;
    PyArrayObject* array = read_weights_from_numpy(file_path, print);
    if (array == nullptr) {
        input.matrix = nullptr;
        return input;
    }

    int ndims = get_numpy_ndims(array);
    get_numpy_shape(array, input, ndims);

    long int size = get_numpy_size(array);
    float* images = convert_PyArrayObject_to_float(array, print, input.shape, ndims);

    float* cuda_images = move_weight_to_cuda(images, size);
    Py_DECREF(array);

    input.ndims = ndims;
    input.size = size;
    input.matrix = cuda_images;

    return input;
}

// Cuda Kernels:

// 11. Matrix Multiplication Kernel:
__global__ void matrixMulKernel(float* matrixA, float* matrixB, float* matrixC, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += matrixA[row * colsA + k] * matrixB[k * colsB + col];
        }
        matrixC[row * colsB + col] = sum;
    }
}

// 12. Matrix Addition Kernel:
__global__ void matrixAddKernel(float* matrixA, float* matrixB, float* matrixC, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;
        matrixC[index] = matrixA[index] + matrixB[index];
    }
}

// 13. Batch Normalization and ReLU Kernel:
__global__ void batchNormReluKernel(float* input, float* output, float* gamma, float* beta, float* mean, float* variance, float epsilon, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;

        // Batch normalization using running mean and variance
        float normalized = (input[index] - mean[col]) / sqrtf(variance[col] + epsilon);
        float bn_output = gamma[col] * normalized + beta[col];

        // ReLU activation
        output[index] = max(0.0f, bn_output);
    }
}

// 14. Softmax Kernel:
__global__ void softmaxKernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int index = row * cols + col;

        // Compute the exponential of each element
        float expVal = expf(input[index]);

        // Compute the sum of exponentials for the row
        float sumExp = 0.0f;
        for (int i = 0; i < cols; ++i) {
            sumExp += expf(input[row * cols + i]);
        }

        // Compute the softmax value for the element
        output[index] = expVal / sumExp;
    }
}

// 15. Matrix Multiplication Main Function:
float* matrixMul(float* matrixA, float* matrixB, int rowsA, int colsA, int rowsB, int colsB){
    float* matrixC;
    cudaMalloc((void **)&matrixC, rowsA * colsB * sizeof(float));
    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);
    matrixMulKernel<<<gridSize, blockSize>>>(matrixA, matrixB, matrixC, rowsA, colsA, colsB);
    return matrixC;
}

// 16. Matrix Addition Main Function:
float* matrixAdd(float* matrixA, float* matrixB, int rows, int cols){
    float* matrixC;
    cudaMalloc((void **)&matrixC, rows * cols * sizeof(float));
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    matrixAddKernel<<<gridSize, blockSize>>>(matrixA, matrixB, matrixC, rows, cols);
    return matrixC;
}

// 17. Batch Normalization and ReLU Main Function:
float* batchNormRelu(float* input, float* gamma, float* beta, float* mean, float* variance, float epsilon, int rows, int cols){
    float* output;
    cudaMalloc((void **)&output, rows * cols * sizeof(float));
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    batchNormReluKernel<<<gridSize, blockSize>>>(input, output, gamma, beta, mean, variance, epsilon, rows, cols);
    return output;
}

// 18. Softmax Main Function:
float* softmax(float* input, int rows, int cols){
    float* matrixC;
    cudaMalloc((void **)&matrixC, rows * cols * sizeof(float));
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    softmaxKernel<<<gridSize, blockSize>>>(input, matrixC, rows, cols);
    return matrixC;
}

// Main function
int main() {
    // Initialize the Python interpreter
    Py_Initialize();
    // Ensure that NumPy is available
    import_array();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Weights fc1_w = read_weights("model/fc1.weight.npy", 0);
    Weights fc2_w = read_weights("model/fc2.weight.npy", 0);
    Weights fc3_w = read_weights("model/fc3.weight.npy", 0);
    Weights fc4_w = read_weights("model/fc4.weight.npy", 0);
    Weights fc1_b = read_weights("model/fc1.bias.npy", 0);
    Weights fc2_b = read_weights("model/fc2.bias.npy", 0);
    Weights fc3_b = read_weights("model/fc3.bias.npy", 0);
    Weights fc4_b = read_weights("model/fc4.bias.npy", 0);

    Weights bn1_gamma = read_weights("model/bn1.weight.npy", 0);
    Weights bn1_beta = read_weights("model/bn1.bias.npy", 0);
    Weights bn1_mean = read_weights("model/bn1_mean.npy", 0);
    Weights bn1_var = read_weights("model/bn1_var.npy", 0);

    Weights bn2_gamma = read_weights("model/bn2.weight.npy", 0);
    Weights bn2_beta = read_weights("model/bn2.bias.npy", 0);
    Weights bn2_mean = read_weights("model/bn2_mean.npy", 0);
    Weights bn2_var = read_weights("model/bn2_var.npy", 0);

    Weights bn3_gamma = read_weights("model/bn3.weight.npy", 0);
    Weights bn3_beta = read_weights("model/bn3.bias.npy", 0);
    Weights bn3_mean = read_weights("model/bn3_mean.npy", 0);
    Weights bn3_var = read_weights("model/bn3_var.npy", 0);

    // Read image
    Inputs image = read_image("images/val_class_16_image.npy", 0);

    // Finalize the Python interpreter
    Py_Finalize();

    // Final matrix of shape 1*17
    int output_row = 1;
    int output_col = 17;

    float* matrixC = nullptr;
    float epsilon = 1e-5;

    // Record the start event
    cudaEventRecord(start, 0);

    matrixC = matrixMul(image.matrix, fc1_w.matrix, image.shape[0], image.shape[1], fc1_w.shape[0], fc1_w.shape[1]);
    matrixC = matrixAdd(matrixC, fc1_b.matrix, fc1_b.shape[0], fc1_b.shape[1]);
    matrixC = batchNormRelu(matrixC, bn1_gamma.matrix, bn1_beta.matrix, bn1_mean.matrix, bn1_var.matrix, epsilon, fc1_b.shape[0], fc1_b.shape[1]);

    matrixC = matrixMul(matrixC, fc2_w.matrix, fc1_b.shape[0], fc1_b.shape[1], fc2_w.shape[0], fc2_w.shape[1]);
    matrixC = matrixAdd(matrixC, fc2_b.matrix, fc2_b.shape[0], fc2_b.shape[1]);
    matrixC = batchNormRelu(matrixC, bn2_gamma.matrix, bn2_beta.matrix, bn2_mean.matrix, bn2_var.matrix, epsilon, fc2_b.shape[0], fc2_b.shape[1]);

    matrixC = matrixMul(matrixC, fc3_w.matrix, fc2_b.shape[0], fc2_b.shape[1], fc3_w.shape[0], fc3_w.shape[1]);
    matrixC = matrixAdd(matrixC, fc3_b.matrix, fc3_b.shape[0], fc3_b.shape[1]);
    matrixC = batchNormRelu(matrixC, bn3_gamma.matrix, bn3_beta.matrix, bn3_mean.matrix, bn3_var.matrix, epsilon, fc3_b.shape[0], fc3_b.shape[1]);

    matrixC = matrixMul(matrixC, fc4_w.matrix, fc3_b.shape[0], fc3_b.shape[1], fc4_w.shape[0], fc4_w.shape[1]);
    matrixC = matrixAdd(matrixC, fc4_b.matrix, fc4_b.shape[0], fc4_b.shape[1]);

    matrixC = softmax(matrixC, fc4_b.shape[0], fc4_b.shape[1]);

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total execution time for all kernels: %f ms", elapsedTime);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float* C = (float *)malloc(output_row * output_col * sizeof(float));
    cudaMemcpy(C, matrixC,  output_row * output_col * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // printing final matrix
    printf("\n#######################\n");
    printf("rowsA %d\n", output_row);
    printf("colsB %d\n", output_col);
    for (int i = 0; i < output_row; i++) {
        for (int j = 0; j < output_col; j++){
            printf("%f ", C[i * output_col + j]);
        }
        printf("\n");
    }

    // Clean up CUDA device memory
    cudaFree(fc1_w.matrix);
    cudaFree(fc2_w.matrix);
    cudaFree(fc3_w.matrix);
    cudaFree(fc4_w.matrix);

    cudaFree(fc1_b.matrix);
    cudaFree(fc2_b.matrix);
    cudaFree(fc3_b.matrix);
    cudaFree(fc4_b.matrix);

    cudaFree(bn1_gamma.matrix);
    cudaFree(bn1_beta.matrix);
    cudaFree(bn1_mean.matrix);
    cudaFree(bn1_var.matrix);

    cudaFree(bn2_gamma.matrix);
    cudaFree(bn2_beta.matrix);
    cudaFree(bn2_mean.matrix);
    cudaFree(bn2_var.matrix);

    cudaFree(bn3_gamma.matrix);
    cudaFree(bn3_beta.matrix);
    cudaFree(bn3_mean.matrix);
    cudaFree(bn3_var.matrix);

    return 0;
}
