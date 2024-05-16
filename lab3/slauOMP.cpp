#include <iostream>
#include <cmath>
#include <omp.h>

const double EPSILON = 0.000001;
const int N = 7000;
const float TAU = 0.00001;
const int MAX_ITERATION_NUM = 10000;

void createMatrix(double* matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            matrix[i * N + j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            if (i == j)
                matrix[i * N + j] = matrix[i * N + j] + N;
        }
        for (int j = 0; j < i; j++)
            matrix[i * N + j] = matrix[j * N + i];
    }
}

void createVector(double* vector) {
    for (int i = 0; i < N; i++)
        vector[i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
}

void setMatrixParts(int* linesPerThread, int* offsets, int size){
    int lines = N / size, rest = N % size, offset = 0;
    for (int i = 0; i < size; ++i) {
        rest ? (linesPerThread[i] = lines + 1, rest--) : linesPerThread[i] = lines;
        offsets[i] = offset;
        offset += linesPerThread[i];
    }
}

double countAccuracy(const double* vector) {
    double accuracy = 0;
    for (int i = 0; i < N; i++)
        accuracy += vector[i] * vector[i];
    return sqrt(accuracy);
}

int main(int argc, char** argv) {
    srand(time(nullptr));

    double begin, end;

    //omp_set_num_threads(4);

    int size = omp_get_max_threads();

    auto* A = new double[N * N];
    auto* x = new double[N];
    auto* x_new = new double[N];
    auto* b = new double[N];

    auto* linesPerThread = new int[size];
    auto* offsets = new int[size];

    int iter_count = 0;
    double accuracy = 1;

    setMatrixParts(linesPerThread, offsets, size);

    begin = omp_get_wtime();

    createMatrix(A);
    createVector(x);
    createVector(b);

    double B_accuracy = countAccuracy(b);
    int i, j;

#pragma omp parallel shared(accuracy) private(i, j)
    {
        int thread_id = omp_get_thread_num();

        int first_index = offsets[thread_id];
        int last_index = first_index + linesPerThread[thread_id];

        while (accuracy > EPSILON && iter_count < MAX_ITERATION_NUM) {
            for (i = first_index; i < last_index; i++) {
                x_new[i] = 0;
                for (j = 0; j < N; j++)
                    x_new[i] += A[i * N + j] * x[j];
            }

            for (i = first_index; i < last_index; i++)
                x_new[i] = x_new[i] - b[i];

#pragma omp single
            accuracy = 0;

#pragma omp for reduction(+: accuracy)
            for (i = 0; i < N; i++)
                accuracy += x_new[i] * x_new[i];

            for (i = first_index; i < last_index; i++) {
                x_new[i] = x[i] - TAU * x_new[i];
                x[i] = x_new[i];
            }

#pragma omp single
            {
                accuracy = sqrt(accuracy) / B_accuracy;
                iter_count++;
            }
        }
    }

    delete[] A;
    delete[] x;
    delete[] x_new;
    delete[] b;
    delete[] linesPerThread;
    delete[] offsets;

    end = omp_get_wtime();

    std::cout << "iter_count = " << iter_count << std::endl;
    std::cout << "total time is " << end - begin << " seconds" << std::endl;

    return 0;
}