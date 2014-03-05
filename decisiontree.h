#ifndef __DECISIONTREE_H__
#define __DECISIONTREE_H__

#include "dtreetypedefs.h"

#include <vector>
#include <unordered_set>

#include "treenode.h"

class DecisionTree{
    public:
        DecisionTree();
        void train(std::vector<InputData> data, std::vector<OutputData> labels);
        //long input_length, samples_length;
        long num_of_classes;
        std::unordered_set<InputValue>* uvalues;

        TreeNode* head;

        int *predict(InputData data);
        double ginii(const std::vector<OutputData> &labels);
        int getNumOfClasses(std::vector<OutputData> labels);
        int *getFreq(std::vector<OutputData> labels);
        TreeNode *buildnode(const std::vector<InputData> &data, const std::vector<OutputData> &labels);
        void calcUniqValues(const std::vector<InputData> &data);
        void divideSet(const std::vector<InputData> &data, 
            const std::vector<OutputData> &labels,
            int col, InputValue value, 
            std::vector<InputData> *s1, std::vector<InputData> *s2,
            std::vector<OutputData> *l1, std::vector<OutputData> *l2);
};

#endif
