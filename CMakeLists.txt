cmake_minimum_required(VERSION 2.8)

SET (CMAKE_CXX_COMPILER  "/usr/bin/clang++")
SET (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-sign-compare -O3 --std=c++11")

find_package( OpenCV REQUIRED )

add_executable(bin/discretize_test discretize_test.cpp discretize.cpp common.cpp)
target_link_libraries(bin/discretize_test ${OpenCV_LIBS})

