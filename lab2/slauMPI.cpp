#include <iostream>
#include <cmath>
#include <mpi.h>

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

void setMatrixParts(int* linesPerProc, int* sendCounts, int* displs, int* offsets, int size){
    int lines = N / size, rest = N % size, offset = 0;
    for (int i = 0; i < size; ++i) {
        rest ? (linesPerProc[i] = lines + 1, rest--) : linesPerProc[i] = lines;
        offsets[i] = offset;
        displs[i] = offset * N;
        sendCounts[i] = linesPerProc[i] * N;
        offset += linesPerProc[i];
    }
}

double countPartAccuracy(const double* vector, int end) {
    double accuracy = 0;
    for (int i = 0; i < end; i++)
        accuracy += vector[i] * vector[i];
    return accuracy;
}

void simpleIteration(const double* A_part, double* b, double* x, const int* linesPerProc, const int* displs, const int* offsets, int rank) {
    auto* x_new = new double[linesPerProc[rank]];
    int iter_count = 0;
    double accuracy = 1;
    double part_accuracy;
    double B_accuracy = sqrt(countPartAccuracy(b, N));

    while (accuracy > EPSILON && iter_count < MAX_ITERATION_NUM) {
        for (int i = 0; i < linesPerProc[rank]; i++) {
            x_new[i] = 0;
            for (int j = 0; j < N; j++)
                x_new[i] += A_part[i * N + j] * x[j];
        }

        for (int i = 0; i < linesPerProc[rank]; i++)
            x_new[i] = x_new[i] - b[offsets[rank] + i];

        part_accuracy = countPartAccuracy(x_new, linesPerProc[rank]);

        for (int i = 0; i < linesPerProc[rank]; i++)
            x_new[i] = x[offsets[rank] + i] - TAU * x_new[i];

        MPI_Allgatherv(x_new, linesPerProc[rank], MPI_DOUBLE,
                       x, linesPerProc, offsets, MPI_DOUBLE, MPI_COMM_WORLD);

        accuracy = 0;

        MPI_Allreduce(&part_accuracy, &accuracy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        accuracy  = sqrt(accuracy) / B_accuracy;
        iter_count++;
    }

    delete[] x_new;
}

int main(int argc, char** argv) {
    srand(time(nullptr));

    int rank, size;
    double begin, end;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto* linesPerProc = new int[size];
    auto* offsets = new int[size];
    auto* sendCounts = new int[size];
    auto* displs = new int[size];
    double* A;

    setMatrixParts(linesPerProc, sendCounts, displs, offsets, size);

    auto* x = new double[N];
    auto* b = new double[N];

    if (!rank) {
        begin = MPI_Wtime();

        A = new double[N * N];
        createMatrix(A);
        createVector(x);
        createVector(b);
    }

    auto* A_part = new double[linesPerProc[rank] * N];

    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A, sendCounts, displs, MPI_DOUBLE,
                 A_part, sendCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    simpleIteration(A_part, b, x, linesPerProc, displs, offsets, rank);

    delete[] A_part;
    delete[] x;
    delete[] b;
    delete[] linesPerProc;
    delete[] sendCounts;
    delete[] displs;
    delete[] offsets;

    if (!rank) {
        delete[] A;
        end = MPI_Wtime();
        std::cout << "total time is " << end - begin << " seconds" << std::endl;
    }

    MPI_Finalize();

    return 0;
}