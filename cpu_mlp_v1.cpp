#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <algorithm>  // Include this header for std::max
#include <chrono>


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

// 7. Function to get shape and allocate in memory:
void get_numpy_shape(PyArrayObject* array, Weights& weights, int ndim){
    npy_intp* shape = PyArray_DIMS(array);
    weights.shape = (int *)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        weights.shape[i] = static_cast<int>(shape[i]);
    }
}

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

    weight.ndims = ndims;
    weight.size = size;
    weight.matrix = matrix;

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

    input.ndims = ndims;
    input.size = size;
    input.matrix = images;

    return input;
}

// CPU Functions:

// 11. Matrix Multiplication Function:
void matrixMul(float* matrixA, float* matrixB, float* matrixC, int rowsA, int colsA, int colsB) {
    for (int row = 0; row < rowsA; ++row) {
        for (int col = 0; col < colsB; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; ++k) {
                sum += matrixA[row * colsA + k] * matrixB[k * colsB + col];
            }
            matrixC[row * colsB + col] = sum;
        }
    }
}

// 12. Matrix Addition Function:
void matrixAdd(float* matrixA, float* matrixB, float* matrixC, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            matrixC[index] = matrixA[index] + matrixB[index];
        }
    }
}

// 13. Batch Normalization and ReLU Function:
void batchNormRelu(float* input, float* output, float* gamma, float* beta, float* mean, float* variance, float epsilon, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            float normalized = (input[index] - mean[col]) / sqrtf(variance[col] + epsilon);
            float bn_output = gamma[col] * normalized + beta[col];
            output[index] = std::max(0.0f, bn_output);
        }
    }
}

// 14. Softmax Function:
void softmax(float* input, float* output, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        float sumExp = 0.0f;
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            float expVal = expf(input[index]);
            sumExp += expVal;
            output[index] = expVal;
        }
        for (int col = 0; col < cols; ++col) {
            int index = row * cols + col;
            output[index] /= sumExp;
        }
    }
}

// Main function
int main() {
    // Initialize the Python interpreter
    Py_Initialize();
    // Ensure that NumPy is available
    import_array();

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

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

    float* matrixC = (float *)malloc(output_row * output_col * sizeof(float));
    float epsilon = 1e-5;

    float* temp1 = (float *)malloc(fc1_b.shape[0] * fc1_b.shape[1] * sizeof(float));
    float* temp2 = (float *)malloc(fc2_b.shape[0] * fc2_b.shape[1] * sizeof(float));
    float* temp3 = (float *)malloc(fc3_b.shape[0] * fc3_b.shape[1] * sizeof(float));

    matrixMul(image.matrix, fc1_w.matrix, temp1, image.shape[0], image.shape[1], fc1_w.shape[1]);
    matrixAdd(temp1, fc1_b.matrix, temp1, fc1_b.shape[0], fc1_b.shape[1]);
    batchNormRelu(temp1, temp1, bn1_gamma.matrix, bn1_beta.matrix, bn1_mean.matrix, bn1_var.matrix, epsilon, fc1_b.shape[0], fc1_b.shape[1]);

    matrixMul(temp1, fc2_w.matrix, temp2, fc1_b.shape[0], fc1_b.shape[1], fc2_w.shape[1]);
    matrixAdd(temp2, fc2_b.matrix, temp2, fc2_b.shape[0], fc2_b.shape[1]);
    batchNormRelu(temp2, temp2, bn2_gamma.matrix, bn2_beta.matrix, bn2_mean.matrix, bn2_var.matrix, epsilon, fc2_b.shape[0], fc2_b.shape[1]);

    matrixMul(temp2, fc3_w.matrix, temp3, fc2_b.shape[0], fc2_b.shape[1], fc3_w.shape[1]);
    matrixAdd(temp3, fc3_b.matrix, temp3, fc3_b.shape[0], fc3_b.shape[1]);
    batchNormRelu(temp3, temp3, bn3_gamma.matrix, bn3_beta.matrix, bn3_mean.matrix, bn3_var.matrix, epsilon, fc3_b.shape[0], fc3_b.shape[1]);

    matrixMul(temp3, fc4_w.matrix, matrixC, fc3_b.shape[0], fc3_b.shape[1], fc4_w.shape[1]);
    matrixAdd(matrixC, fc4_b.matrix, matrixC, fc4_b.shape[0], fc4_b.shape[1]);

    softmax(matrixC, matrixC, fc4_b.shape[0], fc4_b.shape[1]);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // printing final matrix
    printf("rowsA %d\n", output_row);
    printf("colsB %d\n", output_col);
    for (int i = 0; i < output_row; i++) {
        for (int j = 0; j < output_col; j++){
            printf("%f ", matrixC[i * output_col + j]);
        }
        printf("\n");
    }

    // Print execution time
    printf("Execution time: %lf ms\n", elapsed.count());

    // Clean up memory
    free(fc1_w.matrix);
    free(fc2_w.matrix);
    free(fc3_w.matrix);
    free(fc4_w.matrix);

    free(fc1_b.matrix);
    free(fc2_b.matrix);
    free(fc3_b.matrix);
    free(fc4_b.matrix);

    free(bn1_gamma.matrix);
    free(bn1_beta.matrix);
    free(bn1_mean.matrix);
    free(bn1_var.matrix);

    free(bn2_gamma.matrix);
    free(bn2_beta.matrix);
    free(bn2_mean.matrix);
    free(bn2_var.matrix);

    free(bn3_gamma.matrix);
    free(bn3_beta.matrix);
    free(bn3_mean.matrix);
    free(bn3_var.matrix);

    free(temp1);
    free(temp2);
    free(temp3);
    free(matrixC);

    return 0;
}
