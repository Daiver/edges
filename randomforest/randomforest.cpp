#include "randomforest.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

RandomForest::RandomForest(int ansamble_length){
    this->ansamble_length = ansamble_length;
    this->ansamble = new DecisionTree[this->ansamble_length];
    this->indxss = new std::vector<int>[this->ansamble_length];
}

void swap(int *a, int *b){
    int c = *a;
    *a = *b;
    *b = c;
}

std::vector<int> getRandIndxs(int size){
    std::vector<int> indxs(size);
    for(int i = 0; i < size; i++) indxs[i] = i;
    for(int i = 0; i < size; i++) swap(&indxs[i], &indxs[rand() % size]);
    return indxs;
}

InputData resample(InputData data, std::vector<int> indxs){
    InputData res;
    int max_sample_per_set = sqrt(data.size());
    for(int i = 0; i < max_sample_per_set; i++)
        res.push_back(data[indxs[i]]);
    return res;
}

std::vector<InputData> RandomForest::getRandSamples(std::vector<InputData> data, std::vector<int> indxs){
    std::vector<InputData> res(data.size());
    for(int j = 0; j < data.size(); j++)
        res[j] = resample(data[j], indxs);
    return res;
}

void RandomForest::train(std::vector<InputData> data, std::vector<OutputData> label){
    for(int i = 0; i < this->ansamble_length; i++){
        auto indxs = getRandIndxs(data[0].size());
        auto ndata = this->getRandSamples(data, indxs);
        indxss[i] = indxs;
        this->ansamble[i].train(ndata, label);
    }
    this->num_of_classes = this->ansamble[0].num_of_classes;
}

float *RandomForest::predict(InputData sample){
    float *res = new float[num_of_classes];
    memset(res, 0, sizeof(float) * this->num_of_classes);
    
    for(int i = 0; i < this->ansamble_length; i++){
        int *freqs = this->ansamble[i].predict(resample(sample, indxss[i]));
        for(int j = 0; j < this->num_of_classes; j++)
            res[j] += freqs[j];
    }
    float sum = 0;
    for(int j = 0; j < this->num_of_classes; j++) sum += res[j];
    for(int j = 0; j < this->num_of_classes; j++) res[j] /= sum;
    return res;
}
