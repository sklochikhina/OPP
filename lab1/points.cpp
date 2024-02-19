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
            sum += static_cast<long long>(a[i]) * static_cast<long long>(b[j]);

    return sum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int chunk_size = N / size;
    unsigned long long total_res;
    double begin, end;

    if (rank == 0) {
        int* a = new int[N];
        int* b = new int[N];

        createVector(a, b);

        begin = MPI_Wtime();

        for (int i = 1; i < size; i++) {
            MPI_Send(&a[0] + i * chunk_size, chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&b[0], N, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        total_res = mult(a, b, chunk_size);

        unsigned long recv_sum;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&recv_sum, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_res += recv_sum;
        }

        delete[] a;
        delete[] b;
    }
    else {
        int* A_local = new int[chunk_size];
        int* B = new int[N];

        MPI_Recv(&A_local[0], chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&B[0], N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        unsigned long local_sum = mult(A_local, B, chunk_size);

        MPI_Send(&local_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        delete[] A_local;
        delete[] B;
    }

    end = MPI_Wtime();

    if (rank == 0)
        std::cout << "Total result = " << total_res << std::endl
                  << "Total time is " << end - begin << std::endl;

    MPI_Finalize();
    return 0;
}