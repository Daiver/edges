#ifndef __RANDOMFOREST_H__
#define __RANDOMFOREST_H__
#include "decisiontree.h"

#include <vector>
#include "dtreetypedefs.h"

class RandomForest {
    public:
        int ansamble_length;
        DecisionTree *ansamble;
        std::vector<int> *indxss;
        int num_of_classes;
        RandomForest(int ansamble_length);
        void train(std::vector<InputData> data, std::vector<cv::Mat> label);
        std::vector<cv::Mat> predict(InputData sample);
        std::vector<InputData> getRandSamples(std::vector<InputData> data, 
                std::vector<int>);
};

#endif
