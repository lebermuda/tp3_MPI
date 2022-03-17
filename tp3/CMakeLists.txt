cmake_minimum_required (VERSION 2.6)

project (tp3)

add_executable(tp3 main.cpp Matrix.cpp)

INCLUDE(CPack)

# Require MPI for this project:
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
target_link_libraries(tp3 ${MPI_LIBRARIES})