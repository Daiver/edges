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

int **dimSort(const std::vector<std::vector<float>> *data,const std::vector<int> &f_idxs, const std::vector<int> *data_idxs, const std::vector<int> &idxs_old){
    size_t inner_size = f_idxs.size();
    int **idxs = new int*[inner_size];
    for(int fid = 0; fid < inner_size; fid++){
        idxs[fid] = new int[idxs_old.size()];
        memcpy(idxs[fid], &(idxs_old.at(0)), sizeof(idxs_old.at(0)) * idxs_old.size());
        std::sort(idxs[fid], idxs[fid] + idxs_old.size(), sortHandler(data, f_idxs[fid], data_idxs));
    }

    return idxs;
}


