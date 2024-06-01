#include <iostream>
#include <mpi.h>

int N1;
int N2;
int N3;

const int X = 0;
const int Y = 1;

const int NDIMS = 2;

const int n1_mult = 500;
const int n3_mult = 750;

void initNs(const int* dims) {
    N1 = dims[X] * n1_mult;
    N3 = dims[Y] * n3_mult;
    N2 = (N3 + N1) / 2;
}

void generateMatrix(double* matrix, int row, int column) {
    for(int i = 0; i < row; i++)
        for(int j = 0; j < column; j++)
            matrix[i * column + j] = (double)rand() / RAND_MAX * 20.0 - 10.0;
}

void createCartComm(MPI_Comm& cart, int size, int argc, char** argv, int* dims) {
    if (argc <= 2)
        MPI_Dims_create(size, NDIMS, dims);
    else {
        dims[X] = strtol(argv[1], nullptr, 10);
        dims[Y] = strtol(argv[2], nullptr, 10);

        if (dims[X] * dims[Y] != size) exit(EXIT_FAILURE);
    }
    bool reorder = true;
    int periodic[NDIMS] = {};
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periodic, reorder, &cart);
}

void createSubComms(MPI_Comm& cart, MPI_Comm& rows, MPI_Comm& columns) {
    int remain_dims[NDIMS];

    remain_dims[X] = false, remain_dims[Y] = true;
    MPI_Cart_sub(cart, remain_dims, &rows);

    remain_dims[X] = true, remain_dims[Y] = false;
    MPI_Cart_sub(cart, remain_dims, &columns);
}

void mult(double* C_part, const double* A_part, const double* B_part, int A_rows, int B_cols) {
    for (int i = 0; i < A_rows; i++)
        for (int j = 0; j < N2; j++)
            for (int k = 0; k < B_cols; k++)
                C_part[i * B_cols + k] += A_part[i * N2 + j] * B_part[j * B_cols + k];
}

void gatherC(double* C, double* C_part, MPI_Comm& cart, int size, int dim_x, int dim_y) {
    MPI_Datatype C_block, C_blocktype;

    int* recvcounts = new int [size];
    int* displs = new int [size];

    for (int i = 0; i < dim_x; i++)
        for (int j = 0; j < dim_y; j++) {
            recvcounts[i * dim_y + j] = 1;
            displs[i * dim_y + j] = i * n1_mult * dim_y + j;
        }

    MPI_Type_vector(n1_mult, n3_mult, N3, MPI_DOUBLE, &C_block);
    MPI_Type_commit(&C_block);

    MPI_Type_create_resized(C_block, 0, n3_mult * sizeof(double), &C_blocktype);
    MPI_Type_commit(&C_blocktype);

    MPI_Gatherv(C_part, n1_mult * n3_mult, MPI_DOUBLE, C, recvcounts, displs, C_blocktype, 0, MPI_COMM_WORLD);

    MPI_Type_free(&C_block);
    MPI_Type_free(&C_blocktype);

    delete[] displs;
    delete[] recvcounts;
}

int main(int argc, char** argv) {
    int size, rank;
    double start, end;

    double* A;
    double* B;
    double* C;

    double* A_part;
    double* B_part;
    double* C_part;

    MPI_Comm cart;
    MPI_Comm rows;
    MPI_Comm columns;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[NDIMS] = {};
    int coords[NDIMS] = {};

    createCartComm(cart, size, argc, argv, dims);
    initNs(dims);

    createSubComms(cart, rows, columns);

    MPI_Cart_coords(cart, rank, NDIMS, coords);

    if (!coords[X] && !coords[Y]) {
        A = new double [N1 * N2];
        B = new double [N2 * N3];
        C = new double [N1 * N3]{};

        generateMatrix(A, N1, N2);
        generateMatrix(B, N2, N3);
    }

    start = MPI_Wtime();

    int A_part_size = n1_mult * N2;
    int B_part_size = N2 * n3_mult;

    A_part = new double [A_part_size];
    B_part = new double [B_part_size];
    C_part = new double [n1_mult * n3_mult];

    if (!coords[Y])
        MPI_Scatter(A, A_part_size, MPI_DOUBLE, A_part, A_part_size, MPI_DOUBLE, 0, columns);
    MPI_Bcast(A_part, A_part_size, MPI_DOUBLE, 0, rows);

    MPI_Datatype B_block, B_blocktype;

    MPI_Type_vector(N2, n3_mult, N3, MPI_DOUBLE, &B_block);
    MPI_Type_commit(&B_block);

    MPI_Type_create_resized(B_block, 0, n3_mult * sizeof(double), &B_blocktype);
    MPI_Type_commit(&B_blocktype);

    if (!coords[X])
        MPI_Scatter(B, 1, B_blocktype, B_part, n3_mult * N2, MPI_DOUBLE, 0, rows);
    MPI_Bcast(B_part, B_part_size, MPI_DOUBLE, 0, columns);

    mult(C_part, A_part, B_part, n1_mult, n3_mult);

    gatherC(C, C_part, cart, size, dims[X], dims[Y]);

    if (!coords[X] && !coords[Y]) {
        end = MPI_Wtime();
        std::cout << "Time: " << end - start << std::endl;
        delete[] A;
        delete[] B;
        delete[] C;
    }

    delete[] A_part;
    delete[] B_part;
    delete[] C_part;

    MPI_Comm_free(&cart);
    MPI_Comm_free(&rows);
    MPI_Comm_free(&columns);

    MPI_Type_free(&B_block);
    MPI_Type_free(&B_blocktype);

    MPI_Finalize();
    return 0;
}