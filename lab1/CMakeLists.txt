cmake_minimum_required(VERSION 3.27)
project(lab1)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_executable(lab1.1 simple.cpp)
add_executable(lab1.2 points.cpp)
add_executable(lab1.3 collective.cpp)

target_link_libraries(lab1.2 PUBLIC MPI::MPI_CXX)
target_link_libraries(lab1.3 PUBLIC MPI::MPI_CXX)