cmake_minimum_required(VERSION 2.8)

find_package( OpenCV REQUIRED )

add_executable(bin/discretize_test discretize_test.cpp)
target_link_libraries(bin/discretize_test ${OpenCV_LIBS})

