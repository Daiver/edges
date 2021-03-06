cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/FindTBB/CMake/FindTBB/")


SET (CMAKE_CXX_COMPILER  "/usr/bin/clang++")
#SET (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-sign-compare -O3 --std=c++11 ")
SET (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-sign-compare  -O3 --std=c++11  -Wunused-parameter -Wstrict-overflow=5 -Wundef -Wshadow -Wcast-qual -Wcast-align -Wmissing-declarations -Wredundant-decls -Wdisabled-optimization -Woverloaded-virtual -Wctor-dtor-privacy -Wold-style-cast")

find_package( OpenCV REQUIRED )

find_package(TBB)

if(NOT TBB_FOUND)
            MESSAGE (STATUS "TBB not found. Example project will not be built.")
else(NOT TBB_FOUND)
    #add_executable(bin/discretize_test discretize_test.cpp discretize.cpp common.cpp)
    include_directories(${TBB_INCLUDE_DIRS})
    add_executable(bin/test test.cpp 
        common.cpp decisiontree.cpp treenode.cpp
        discretize.cpp desc.cpp randomforest.cpp 
        gradientMex.cpp
        )
    add_executable(bin/discretize_test discretize_test.cpp 
        common.cpp decisiontree.cpp treenode.cpp
        discretize.cpp 
        gradientMex.cpp
        )
    add_executable(bin/train train.cpp 
        common.cpp decisiontree.cpp treenode.cpp
        discretize.cpp 
        randomforest.cpp desc.cpp 
        gradientMex.cpp
        )
    add_executable(bin/activate activate.cpp 
        common.cpp decisiontree.cpp treenode.cpp
        discretize.cpp 
        randomforest.cpp desc.cpp 
        gradientMex.cpp
        )

    add_executable(bin/gradTest gradMexTest.cpp 
        common.cpp decisiontree.cpp treenode.cpp
        discretize.cpp 
        randomforest.cpp desc.cpp 
        gradientMex.cpp
        )

    target_link_libraries(bin/discretize_test ${OpenCV_LIBS} tbb)
    target_link_libraries(bin/gradTest ${OpenCV_LIBS} tbb)
    target_link_libraries(bin/test ${OpenCV_LIBS} tbb)
    target_link_libraries(bin/train ${OpenCV_LIBS} tbb)
    target_link_libraries(bin/activate ${OpenCV_LIBS} tbb)
    #target_link_libraries(bin/test2 ${TBB_LIBS})
    link_directories(${TBB_LIBRARY_DIRS})
    #add_executable(FindTBB_example ${FindTBB_example_SRCS_CPP})

endif(NOT TBB_FOUND)

