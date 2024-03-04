#include <iostream>
#include <cmath>

const double EPSILON = 0.0001;
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
    for (int i = 0; i < N; i++)
        accuracy += vector[i] * vector[i];
    return sqrt(accuracy);
}

void simpleIteration(const double* A, double* b, double* x) {
    auto* x_new = new double[N];
    int iter_count = 0;
    double accuracy = 1;

    while (accuracy > EPSILON && iter_count < MAX_ITERATION_NUM) {
        for (int i = 0; i < N; i++) {
            x_new[i] = 0;
            for (int j = 0; j < N; j++)
                x_new[i] += A[i * N + j] * x[j];
        }

        for (int i = 0; i < N; i++)
            x_new[i] = x_new[i] - b[i];

        accuracy = countAccuracy(x_new) / countAccuracy(b);

        for (int i = 0; i < N; i++)
            x[i] = x[i] - TAU * x_new[i];

        iter_count++;
    }

    delete[] x_new;
}

int main() {
    srand(time(nullptr));

    auto* A = new double[N * N];
    auto* x = new double[N];
    auto* b = new double[N];

    createMatrix(A);
    createVector(x);
    createVector(b);

    time_t begin = time(nullptr);

    simpleIteration(A, b, x);

    time_t end = time(nullptr);

    std::cout << "total time is " << end - begin << " seconds" << std::endl;

    delete[] A;
    delete[] x;
    delete[] b;

    return 0;
}