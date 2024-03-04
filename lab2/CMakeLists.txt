cmake_minimum_required(VERSION 3.27)
project(lab2)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(lab2.1 slau.cpp)
add_executable(lab2.2 slauMPI.cpp)

target_link_libraries(lab2.2 PUBLIC MPI::MPI_CXX)