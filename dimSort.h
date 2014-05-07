#include "defines.h"

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <string.h>


class sortHandler{
    public:
    int idx;
    const std::vector<std::vector<float>> *data;
    const std::vector<int> *idxs;
    sortHandler(const std::vector<std::vector<float>> *data, int idx,
            const std::vector<int> *idxs){
        this->data = data;
        this->idxs = idxs;
        this->idx = idx;
    }

    bool operator() (int a, int b){
        return this->data->at(this->idxs->at(a))[this->idx] 
            < this->data->at(this->idxs->at(b))[this->idx];
    }
} ;

static int **dimSort(
        const std::vector<std::vector<float>> *data,
        const std::vector<int> &f_idxs, 
        const std::vector<int> *data_idxs
        //const std::vector<int> &idxs_old
        ){
    size_t inner_size = f_idxs.size();
    int **idxs = new int*[inner_size];
#ifdef SORT_DEBUG
    printf("In sort inner_size %d\n", inner_size);
#endif
    for(int fid = 0; fid < inner_size; fid++){
        idxs[fid] = new int[data_idxs->size()];
        for(int i = 0; i < data_idxs->size(); i++)
            idxs[fid][i] = i;
        //memcpy(idxs[fid], &(idxs_old.at(0)), sizeof(idxs_old.at(0)) * idxs_old.size());
#ifdef SORT_DEBUG
        printf("Sorting # %di %d\n", fid, f_idxs[fid]);
#endif
        std::sort(idxs[fid], 
                idxs[fid] + data_idxs->size(), 
                sortHandler(data, f_idxs[fid], data_idxs));
    }

    return idxs;
}
/*
int main(){
    std::vector<std::vector<float>> data = {
        {10, 15, 0, 4},//0
        {7, 5, 70,  0},//1
        {2, 5, 10,  5},//2
        {1, 12, 0,  2},//3
        {1, 13, 0,  1},//4
        {5, 11, 20, 3} //5
    };
    std::vector<int> data_idxs = {1, 3, 5};
    std::vector<int> f_idxs    = {0, 2};
    std::vector<int> to_sort  = {0, 1, 2}; 
    int **res = dimSort(&data, f_idxs, &data_idxs, to_sort);
    for(int f = 0; f < f_idxs.size(); f++){
        for(int i = 0; i < to_sort.size(); i++){
            printf("%d ", res[f][i]);
        }
        printf("\n");
    }
}
*/
