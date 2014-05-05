#include "defines.h"

#include "randomforest.h"
#include "common.h"
#include "desc.h"
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
    //this->indxss = new std::vector<int>[this->ansamble_length];
}

void RandomForest::save(const char *fname){
    for(int i = 0; i < this->ansamble_length; i++){
        char name[128];
        sprintf(name, "%s_%d.tree", fname, i);
        this->ansamble[i].save(name);
    }
}

void RandomForest::load(const char *fname){
    for(int i = 0; i < this->ansamble_length; i++){
        char name[128];
        sprintf(name, "%s_%d.tree", fname, i);
        this->ansamble[i].load(name);
    }
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

void RandomForest::train_one_tree(const std::vector<InputData>&, const std::vector<cv::Mat>& , int tree_num){
    //int frame_size = data.size() ;
    int i = tree_num;
    printf("Tree num %d\n", i);
    std::vector<cv::Mat> images, gtruth;
    read_imgList2("images6.txt", &images, &gtruth);

    std::vector<cv::Mat> img_patches, gt_patches;
    //std::vector<std::vector<float>> data(img_patches.size());
    /*for(int i = 0; i < images.size(); i++){
        cutPatchesFromImage3(images[i], gtruth[i], &img_patches, &gt_patches, 1500, 1500);
    }
    for(int i = 0; i < data.size(); i++){
        patchesToVec(img_patches[i], &data[i]);
    }*/
    //srand( time(NULL) );
    std::vector<std::vector<float>> data;
    for(int i = 0; i < images.size(); i++){
        std::vector<cv::Mat> chnReg, chnSim;
        imageChns(images[i], &chnReg, &chnSim);
        chnsToVecs(chnReg, chnSim, images[i], gtruth[i], &data, &gt_patches, 100, 100);
    }
   
    printf("dataset size: %d\n", data.size());
    printf("features len: %d\n", data[0].size());
    std::vector<int> data_idx;
    for(int i = 0; i < data.size();i++){
        data_idx.push_back(i);
    }
    this->ansamble[i].train(&data, data_idx, gt_patches);
    printf("=======FINISH Tree num %d========\n", i);
    //this->ansamble[i].train(data, label);

}

void RandomForest::train(std::vector<InputData> &data, std::vector<cv::Mat> &label){
#ifdef ENABLE_TBB
    tbb::task_scheduler_init init_object(4);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, this->ansamble_length) , 
            [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i){
                this->train_one_tree(data, label, i);
            }
    });
#else

    for(int i = 0; i < this->ansamble_length; i++){
        this->train_one_tree(data, label, i);
    }
#endif
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
