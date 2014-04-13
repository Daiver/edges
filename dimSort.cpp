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

int **dimSort(const std::vector<std::vector<float>> *data,
        const std::vector<int> &f_idxs, 
        const std::vector<int> *data_idxs, 
        const std::vector<int> &idxs_old){
    size_t inner_size = f_idxs.size();
    int **idxs = new int*[inner_size];
    for(int fid = 0; fid < inner_size; fid++){
        idxs[fid] = new int[idxs_old.size()];
        memcpy(idxs[fid], &(idxs_old.at(0)), sizeof(idxs_old.at(0)) * idxs_old.size());
        std::sort(idxs[fid], 
                idxs[fid] + idxs_old.size(), 
                sortHandler(data, f_idxs[fid], data_idxs));
    }

    return idxs;
}

int main(){
    std::vector<std::vector<float>> data = {
        {10, 15, 0, 4},//0
        {7, 5, 70,  0},//1 7 70
        {2, 5, 10,  5},//2
        {1, 12, 0,  2},//3 1 0
        {1, 13, 0,  1},//4
        {5, 11, 200, 3} //5 5 20
    };
    std::vector<int> data_idx = {1, 3, 5};//3 5 1
    std::vector<int> f_idxs    = {0, 2};
    std::vector<int> to_sort  = {0, 1, 2}; 
    int **idxs = dimSort(&data, f_idxs, &data_idx, to_sort);
    /*for(int f = 0; f < f_idxs.size(); f++){
        for(int i = 0; i < to_sort.size(); i++){
            printf("%d ", res[f][i]);
        }
        printf("\n");
    }*/
    for(int f = 0; f < f_idxs.size(); f++){
        int val0 = data.at(data_idx[idxs[0][f]])[f_idxs[f]];
        for(int i = 1; i < data_idx.size(); i++){
            int id = data_idx[idxs[f][i]];
            int val = data.at(id)[f_idxs[f]];
            if (val0 > val) printf("ERRRRR\n");
            val0 = val;
        }
    }

}
