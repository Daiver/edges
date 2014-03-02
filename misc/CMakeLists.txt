cmake_minimum_required(VERSION 2.8)

find_package( OpenCV REQUIRED )

add_executable(pca_test pca_test.cpp)
target_link_libraries(pca_test ${OpenCV_LIBS})
