#ifndef __DECISIONTREE_H__
#define __DECISIONTREE_H__

#include "dtreetypedefs.h"

#include <vector>
#include <unordered_set>
#include "discretize.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "treenode.h"

class DecisionTree{
    public:
        DecisionTree();
        void train(std::vector<InputData> data, std::vector<cv::Mat> labels);
        //long input_length, samples_length;
        //long num_of_classes;
        std::unordered_set<InputValue>* uvalues;

        TreeNode* head;

        cv::Mat predict(InputData data);
        double ginii(const std::vector<OutputData> &labels, int num_of_classes);
        int getNumOfClasses(std::vector<OutputData> labels);
        int *getFreq(std::vector<OutputData> labels, int num_of_classes);
        TreeNode *buildnode(const std::vector<InputData> &data, 
                const std::vector<cv::Mat>& labels);
        void calcUniqValues(const std::vector<InputData> &data);
        void divideSet(const std::vector<InputData> &data, const std::vector<OutputData> &labels,
            const std::vector<cv::Mat> &seg,
            int col, InputValue value, 
            std::vector<InputData> *s1, std::vector<InputData> *s2,
            std::vector<OutputData> *l1, std::vector<OutputData> *l2,
            std::vector<cv::Mat> *g1, std::vector<cv::Mat> *g2,
            std::vector<int> *i1, std::vector<int> *i2);
};

#endif
