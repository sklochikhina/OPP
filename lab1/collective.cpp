#include <iostream>
#include <mpi.h>

const int N = 100032;

void createVector(int* a, int* b) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
}

unsigned long long mult(int* a, int* b, const int a_size) {
    unsigned long long sum = 0;

    for (int i = 0; i < a_size; i++)
        for (int j = 0; j < N; j++)
            sum += static_cast<unsigned long long>(a[i]) * static_cast<unsigned long long>(b[j]);

    return sum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned long long local_res = 0, total_res;

    int* a;
    int* b = new int[N];

    const int chunk_size = N / size;
    double begin, end;

    if (rank == 0) {
        a = new int[N];
        createVector(a, b);

        total_res = 0;

        begin = MPI_Wtime();
    }

    int* a_part = new int[chunk_size];

    MPI_Bcast(&b[0], N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(&a[0], chunk_size, MPI_INT, &a_part[0], chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    local_res = mult(a_part, b, chunk_size);

    MPI_Reduce(&local_res, &total_res, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        end = MPI_Wtime();
        std::cout << "Total result = " << total_res << std::endl
                  << "Total time is " << end - begin << std::endl;

        delete[] a;
        delete[] b;
        delete[] a_part;
    }

    MPI_Finalize();

    return 0;
}