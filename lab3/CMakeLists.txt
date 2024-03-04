cmake_minimum_required(VERSION 3.27)
project(lab3)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(lab3.1 slau.cpp)
add_executable(lab3.2 slauOMP.cpp)

target_link_libraries(lab3.2 PUBLIC MPI::MPI_CXX)