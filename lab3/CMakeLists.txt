cmake_minimum_required(VERSION 3.27)
project(lab3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

find_package(OpenMP REQUIRED)

add_executable(lab3.1 slauOMP.cpp)
add_executable(lab3.2 schedule.cpp)

target_link_libraries(lab3.1 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(lab3.2 PUBLIC OpenMP::OpenMP_CXX)