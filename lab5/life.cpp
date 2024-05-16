#include <iostream>
#include <mpi.h>

const int MAX_ITERATIONS_COUNT = 5000;
const int N = 400;

const bool ALIVE = true, DEAD = false;

void initializeCells(bool* cells) {
    cells[0 * N + 1] = cells[1 * N + 2] = cells[2 * N + 0] = cells[2 * N + 1] = cells[2 * N + 2] = ALIVE;
    /*for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cells[i * N + j] = (rand() % 2) == 1;*/
}

void setMatrixParts(int* linesPerProc, int* sendCounts, int* displs, int* offsets, int size){
    int lines = N / size, rest = N % size, offset = 0;
    for (int i = 0; i < size; i++) {
        rest ? (linesPerProc[i] = lines + 1, rest--) : linesPerProc[i] = lines;
        offsets[i] = offset;
        displs[i] = offset * N;
        sendCounts[i] = linesPerProc[i] * N;
        offset += linesPerProc[i];
    }
}

int getIndex(int index, int size) {
    if (index < 0)
        return size - 1;
    else if (index >= size)
        return 0;
    else
        return index;
}

int getLiveNeighborsAmount(const bool* prev_cells, int i, int j) {
    int live_neighbors = 0;
    for (int y = -1; y <= 1; y++) {
        int ni = getIndex(i + y, N + 2);

        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0)
                continue;
            int nj = getIndex(j + x, N);
            if (prev_cells[ni * N + nj])
                live_neighbors++;
        }
    }
    return live_neighbors;
}

bool updateCell(const bool* prev_cells, int i, int j){
    int live_neighbors = getLiveNeighborsAmount(prev_cells, i, j);

    if (prev_cells[i * N + j])
        if (live_neighbors < 2 || live_neighbors > 3)
            return DEAD;
        else
            return ALIVE;
    else
    if (live_neighbors == 3)
        return ALIVE;
    else
        return DEAD;
}

void updateInnerCells(bool* part, const bool* prev_cells, int linesProc) {
    int end = linesProc + 2;

    for (int i = 2; i < end; i++)
        for (int j = 0; j < N; j++)
            part[i * N + j] = updateCell(prev_cells, i, j);
}

void updateUpOrDownCells(bool* part, const bool* prev_cells, int i) {
    for (int j = 0; j < N; j++)
        part[i * N + j] = updateCell(prev_cells, i, j);
}

bool isEqual(const bool* part, const bool* prev_cells, int part_size){
    for(int i = N; i < part_size - N; i++)
        if (part[i] != prev_cells[i])
            return false;
    return true;
}

void getVectorOfFlags(bool** generations, const bool* part, bool* process_flags, int iter, int num_procs, int part_size) {
    for (int i = 0; i < iter; i++)
        process_flags[i] = isEqual(part, generations[i], part_size);
    for (int i = 1; i < num_procs; i++)
        for (int j = 0; j < iter; j++)
            process_flags[i * iter + j] = process_flags[0 * iter + j];
}

int checkFlags(const bool* flags, int sizeOfRow, int num_procs) {
    if (num_procs == 1) {
        for (int column = 0; column < sizeOfRow; column++)
            if (flags[column])
                return ++column;
    }
    else
        for(int column = 0; column < sizeOfRow; column++)
            for (int row = 1; row < num_procs; row++) {
                if (flags[column] != flags[row * sizeOfRow + column])
                    break;
                else
                if (row == num_procs - 1 && flags[column])
                    return ++column;
            }
    return -1;
}

int main(int argc, char** argv) {
    srand(time(nullptr));

    int rank, size;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool* cells = nullptr;
    bool* part;
    bool** generations;

    int* linesPerProc;
    int* offsets;
    int* sendCounts;
    int* displs;

    linesPerProc = new int[size];
    offsets = new int[size];
    sendCounts = new int[size];
    displs = new int[size];

    setMatrixParts(linesPerProc, sendCounts, displs, offsets, size);

    int part_size = sendCounts[rank] + 2 * N;
    part = new bool[part_size];

    if (!rank) {
        cells = new bool[N * N]();
        initializeCells(cells);
    }

    MPI_Scatterv(cells, sendCounts, displs, MPI_C_BOOL,
                 &part[N], sendCounts[rank], MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (!rank) delete[] cells;

    delete[] offsets;
    delete[] sendCounts;
    delete[] displs;

    generations = new bool*[MAX_ITERATIONS_COUNT]{nullptr};

    int iter_equal, iter_count;

    start = MPI_Wtime();

    for (iter_count = 1; iter_count < MAX_ITERATIONS_COUNT; iter_count++) {
        generations[iter_count - 1] = part;

        //if (!rank) printf("iter = %d\n", iter_count);

        MPI_Request send_request[3];
        MPI_Request recv_request[2];

        MPI_Isend(&part[N], N, MPI_C_BOOL, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &send_request[0]);
        MPI_Isend(&part[part_size - 2 * N], N, MPI_C_BOOL, (rank + 1) % size, 1, MPI_COMM_WORLD, &send_request[1]);

        part = new bool[part_size];

        MPI_Irecv(&generations[iter_count - 1][0], N, MPI_C_BOOL, (rank - 1 + size) % size, 1, MPI_COMM_WORLD, &recv_request[0]);
        MPI_Irecv(&generations[iter_count - 1][part_size - N], N, MPI_C_BOOL, (rank + 1) % size, 0, MPI_COMM_WORLD, &recv_request[1]);

        bool process_flags[size * (iter_count - 1)];
        bool all_flags[size * (iter_count - 1)];

        if (iter_count > 1) {
            getVectorOfFlags(generations, generations[iter_count - 1], process_flags, iter_count - 1, size, part_size);
            MPI_Ialltoall(process_flags, iter_count - 1, MPI_C_BOOL,
                          all_flags, iter_count - 1, MPI_C_BOOL, MPI_COMM_WORLD, &send_request[2]);
        }

        // inner lines
        updateInnerCells(part, generations[iter_count - 1], linesPerProc[rank] - 2);

        // first line
        MPI_Wait(&recv_request[0], MPI_STATUS_IGNORE);
        updateUpOrDownCells(part, generations[iter_count - 1], 1);

        // last line
        MPI_Wait(&recv_request[1], MPI_STATUS_IGNORE);
        updateUpOrDownCells(part, generations[iter_count - 1], part_size / N - 2);

        MPI_Cancel(&send_request[0]);
        MPI_Cancel(&send_request[1]);

        if (iter_count > 1) {
            MPI_Wait(&send_request[2], MPI_STATUS_IGNORE);
            iter_equal = checkFlags(all_flags, iter_count - 1, size);

            if (iter_equal != -1) {
                iter_count--;
                break;
            }
        }
    }

    if (!rank) {
        end = MPI_Wtime();
        std::cout << "Time: " << end - start << std::endl;
        std::cout << "Iteration count: " << iter_count << std::endl;
        std::cout << "Match on iteration: " << iter_equal << std::endl;
    }

    delete[] linesPerProc;

    delete[] part;
    for (int i = 0; generations[i] != nullptr; i++)
        delete[] generations[i];
    delete[] generations;

    MPI_Finalize();

    return 0;
}
