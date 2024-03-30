#include <iostream>
#include <cmath>
#include <omp.h>

const double EPSILON = 0.000001;
const int N = 4000;
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

double countAccuracy(const double* vector) {
    double accuracy = 0;

#pragma omp parallel for reduction(+ : accuracy) schedule(runtime)
    for (int i = 0; i < N; i++)
        accuracy += vector[i] * vector[i];

    return sqrt(accuracy);
}

int main(int argc, char** argv) {
    srand(time(nullptr));

    double begin, end;

    auto* A = new double[N * N];
    auto* x = new double[N];
    auto* x_new = new double[N];
    auto* b = new double[N];

    int iter_count = 0;
    double accuracy = 1;

    begin = omp_get_wtime();

    createMatrix(A);
    createVector(x);
    createVector(b);

    double B_accuracy = countAccuracy(b);

    while (accuracy > EPSILON && iter_count < MAX_ITERATION_NUM) {

#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++) {
            x_new[i] = 0;
            for (int j = 0; j < N; j++)
                x_new[i] += A[i * N + j] * x[j];
        }

#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++)
            x_new[i] = x_new[i] - b[i];

        accuracy = countAccuracy(x_new) / B_accuracy;

#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < N; i++) {
            x_new[i] = x[i] - TAU * x_new[i];
            x[i] = x_new[i];
        }

        iter_count++;
    }

    delete[] A;
    delete[] x;
    delete[] x_new;
    delete[] b;

    end = omp_get_wtime();

    std::cout << "iter_count = " << iter_count << std::endl;
    std::cout << "total time is " << end - begin << " seconds" << std::endl;

    return 0;
}