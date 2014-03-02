SET (CMAKE_CXX_COMPILER  "/usr/bin/clang++")
SET (CMAKE_CXX_FLAGS "-Wall --std=c++11")

add_executable(bin/main main.cpp decisiontree.cpp treenode.cpp randomforest.cpp)
add_executable(bin/spam spam.cpp decisiontree.cpp treenode.cpp randomforest.cpp)
add_executable(bin/wine wine.cpp decisiontree.cpp treenode.cpp randomforest.cpp)
