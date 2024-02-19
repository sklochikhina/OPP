#include <iostream>
#include <vector>

const int N = 100032;

void createVector(int* a, int* b) {
    srand(time(NULL));
    for (long long int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
}

long long int mult(int* a, int* b) {
    long long sum = 0;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += static_cast<long long>(a[i]) * static_cast<long long>(b[j]);

    return sum;
}

int main(int argc, char** argv) {

    int* a = new int[N];
    int* b = new int[N];

    createVector(a, b);

    time_t begin = time(NULL);
    std::cout << "Total result = " << mult(a, b) << std::endl;
    time_t end = time(NULL);

    std::cout << "Total time is " << end - begin << " seconds" << std::endl;

    delete[] a;
    delete[] b;

    return 0;
}