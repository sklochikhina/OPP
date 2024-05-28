#include <iostream>
#include <pthread.h>
#include <mpi.h>
#include <deque>
#include <valarray>

std::deque<int> tasks;
pthread_mutex_t mutex;
pthread_cond_t criticalCond;

bool isEnough = false;
const int L = 10000;
const int ITER_COUNT = 4;

const int TASKS_PER_PROC = 400;
int CRITICAL = TASKS_PER_PROC / 10;
const int NO_TASKS = -1;
int MULTIPLIER = TASKS_PER_PROC;

double globalRes = 0;
int completedTasks = 0;

int size, rank;

void printResult(int, int&, double, double, double&);

int countWeight(int iterCounter, int idx) {
    return abs(50 - (idx % TASKS_PER_PROC)) * abs(rank - (iterCounter % size)) * L;
}

int getTask() {
    int tmp = tasks.front();
    tasks.pop_front();
    return tmp;
}

void doWork(int repeatNum) {
    double tempRes = 0;
    for (int i = 0; i < repeatNum; i++)
        tempRes += std::sin(i);
    globalRes += tempRes;
    completedTasks++;
}

bool isAllWorkDone(const int* extraTasks) {
    for (int i = 0; i < size; i++)
        if (extraTasks[i] != NO_TASKS)
            return false;
    return true;
}

void* workerThread(void* me) {
    int repeatNum;
    while (true) {
        pthread_mutex_lock(&mutex);
        if (!tasks.empty()) repeatNum = getTask();
        else repeatNum = NO_TASKS;
        pthread_mutex_unlock(&mutex);

        if (repeatNum != NO_TASKS) doWork(repeatNum);

        pthread_mutex_lock(&mutex);
        if (isEnough) {
            while (!tasks.empty()) {
                //printf("rank %d worker left tasks = %zu\n", rank, tasks.size());
                repeatNum = getTask();
                doWork(repeatNum);
            }
            pthread_mutex_unlock(&mutex);
            break;
        }
        if (tasks.size() <= CRITICAL) pthread_cond_signal(&criticalCond);
        pthread_mutex_unlock(&mutex);
    }
    return nullptr;
}

void* managerThread(void* me) {
    while (true) {
        int needWorkRank;
        MPI_Recv(&needWorkRank, 1, MPI_INT, MPI_ANY_SOURCE, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (needWorkRank == NO_TASKS) break;

        int extraTask;
        
        pthread_mutex_lock(&mutex);
        if (!tasks.empty() && tasks.size() > CRITICAL) {
            extraTask = tasks.back();
            tasks.pop_back();
        }
        else extraTask = NO_TASKS;
        pthread_mutex_unlock(&mutex);

        MPI_Send(&extraTask, 1, MPI_INT, needWorkRank, rank, MPI_COMM_WORLD);
    }
    return nullptr;
}

void* requesterThread(void* me) {
    while (true) {
        pthread_mutex_lock(&mutex);
        pthread_cond_wait(&criticalCond, &mutex);
        pthread_mutex_unlock(&mutex);

        int needWorkRank = rank;

        for (int i = 0; i < size; i++)
            if (i != rank)
                MPI_Send(&needWorkRank, 1, MPI_INT, i, i, MPI_COMM_WORLD);

        int* extraTasks = new int[size];

        for (int i = 0; i < size; i++) {
            if (i == rank) {
                extraTasks[i] = NO_TASKS;
                continue;
            }
            MPI_Recv(&extraTasks[i], 1, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (isAllWorkDone(extraTasks)) {
            delete[] extraTasks;
            MPI_Barrier(MPI_COMM_WORLD);
            break;
        }

        pthread_mutex_lock(&mutex);
        for (int i = 0; i < size; i++)
            if (extraTasks[i] != NO_TASKS)
                tasks.push_back(extraTasks[i]);
        delete[] extraTasks;
        pthread_mutex_unlock(&mutex);
    }

    pthread_mutex_lock(&mutex);
    isEnough = true;
    MPI_Send(&NO_TASKS, 1, MPI_INT, rank, rank, MPI_COMM_WORLD);
    pthread_mutex_unlock(&mutex);

    return nullptr;
}

int main(int argc, char** argv) {
    int proc_rank, procs_num;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, nullptr);

    MPI_Comm_size(MPI_COMM_WORLD, &procs_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    rank = proc_rank;
    size = procs_num;

    pthread_attr_t attrs;

    pthread_attr_init(&attrs);
    pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);

    pthread_mutex_init(&mutex, nullptr);
    pthread_cond_init(&criticalCond, nullptr);

    double startTime, endTime;

    int start = TASKS_PER_PROC * rank;
    int end = start + TASKS_PER_PROC;

    MULTIPLIER /= size * 2;

    double maxImbalanceProportion = 0;

    if (!rank) startTime = MPI_Wtime();

    for (int iterCounter = 0; iterCounter < ITER_COUNT; iterCounter++) {
        for (int j = start; j < end; j++)
            tasks.push_back(countWeight(iterCounter, j));

        isEnough = false;

        pthread_t worker;
        pthread_t manager;
        pthread_t requester;

        double iterStartTime = MPI_Wtime();

        pthread_create(&worker, &attrs, workerThread, nullptr);
        pthread_create(&manager, &attrs, managerThread, nullptr);
        pthread_create(&requester, &attrs, requesterThread, nullptr);

        pthread_join(worker, nullptr);
        pthread_join(manager, nullptr);
        pthread_join(requester, nullptr);

        double iterEndTime = MPI_Wtime();
        
        printResult(iterCounter, completedTasks, iterEndTime, iterStartTime, maxImbalanceProportion);

        globalRes = 0;
        completedTasks = 0;

        MPI_Barrier(MPI_COMM_WORLD);
    }

    pthread_attr_destroy(&attrs);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&criticalCond);

    MPI_Barrier(MPI_COMM_WORLD);

    if (!rank) {
        endTime = MPI_Wtime();
        std::cout << "Time: " << endTime - startTime << std::endl;
        std::cout << "Max imbalance proportion: " << maxImbalanceProportion << std::endl;
    }

    MPI_Finalize();

    return 0;
}

void printResult(int iterCounter, int& completed, double iterEndTime, double iterStartTime, double& maxImbalanceProportion) {
    double iterTime = iterEndTime - iterStartTime;

    for (int j = 0; j < size; j++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == j) {
            std::cout << "Iter " << iterCounter << ", rank " << rank << ":" << std::endl;
            std::cout << "\tTasks computed: " << completed << "\n\tGlobal result: " << globalRes << "\n\tTime for iteration: " << iterTime << std::endl;
            std::cout << "-----------------------------------------------------" << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double maxIterationTime, minIterationTime;

    MPI_Reduce(&iterTime, &maxIterationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&iterTime, &minIterationTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (!rank) {
        double imbalanceTime = maxIterationTime - minIterationTime;
        double imbalanceProportion = (imbalanceTime / maxIterationTime) * 100.0;
        if (imbalanceProportion > maxImbalanceProportion)
            maxImbalanceProportion = imbalanceProportion;
        std::cout << "| Iter " << iterCounter << " | Imbalance time: " << imbalanceTime << " | Imbalance proportion: " << imbalanceProportion << " |" << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}