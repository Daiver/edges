#include "randomforest.h"
#include "discretize.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"


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

void RandomForest::train_one_tree(const std::vector<InputData>& data, const std::vector<cv::Mat>& label, int tree_num){
    int frame_size = data.size() ;
    int i = tree_num;
    printf("Tree num %d\n", i);
    //auto indxs = getRandIndxs(data[0].size());
    //auto ndata = this->getRandSamples(data, indxs);
    //indxss[i] = indxs;
    //std::vector<InputData> n_data_idx; //(data.size()/this->ansamble_length);
    std::vector<int> n_data_idx; //(data.size()/this->ansamble_length);
    std::vector<cv::Mat> n_labels; //(data.size()/this->ansamble_length);
    int neg_size = 0;
    for(int j = 0; j < frame_size; j++){
        int indx = rand() % data.size();
        //printf("indx %d\n", indx);
        cv::Scalar mean, std;
        cv::meanStdDev(label[indx], mean, std);
        //printf("%f\n", std[0]);
        if(std[0] == 0.0) {
            neg_size++;
            if (neg_size > (data.size()/2)) {j--; continue;}
        }
        n_data_idx.push_back(indx);
        n_labels.push_back(label[indx]);
    }
    /*for(int j = (frame_size/2) * (i); 
            j < (frame_size/2)*(i + 1); j++){
        if (j >= data.size()) continue;
        n_data.push_back(data[j]);
        n_labels.push_back(label[j]);
    }
    printf("%d: tree dt size %d\n", i, n_data.size());*/
    this->ansamble[i].train(&data, n_data_idx, n_labels);
    printf("=======FINISH Tree num %d========\n", i);
    //this->ansamble[i].train(data, label);

}

void RandomForest::train(std::vector<InputData> data, std::vector<cv::Mat> label){
    tbb::task_scheduler_init init_object(4);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, this->ansamble_length) , 
            [=](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i){
                this->train_one_tree(data, label, i);
            }
    });

    /*for(int i = 0; i < this->ansamble_length; i++){
        this->train_one_tree(data, label, i);
    }*/
    //this->num_of_classes = this->ansamble[0].num_of_classes;
}

std::vector<cv::Mat> RandomForest::predict(InputData sample){
    //float *res = new float[num_of_classes];
    //memset(res, 0, sizeof(float) * this->num_of_classes);
    std::vector<cv::Mat> res;
    for(int i = 0; i < this->ansamble_length; i++){
        cv::Mat r = this->ansamble[i].predict(sample);
        res.push_back(r);
        //for(int j = 0; j < this->num_of_classes; j++)
        //    res[j] += freqs[j];
    }
    //float sum = 0;
    //for(int j = 0; j < this->num_of_classes; j++) sum += res[j];
    //for(int j = 0; j < this->num_of_classes; j++) res[j] /= sum;
    int num_of_classes, seg_idx;
    std::vector<int> labels(res.size(), 0);
    selectFeaturesFromPatches(res, &labels, &num_of_classes, &seg_idx);
    res.push_back(res[seg_idx]);
    return res;
}
