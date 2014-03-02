#include <iostream>

template <class T>
void printV(T vec, size_t size){
        for(size_t i = 0 ; i < size; i++){
                    std::cout << vec[i] << " ";
                        }
            std::cout << std::endl;
}

