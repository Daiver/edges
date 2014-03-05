#include "decisiontree.h"

#include<stdio.h>

#include <string.h>
#include <vector>
#include <unordered_set>

void DecisionTree::train(std::vector<InputData> data, std::vector<OutputData> labels){
    this->num_of_classes = getNumOfClasses(labels);
    this->calcUniqValues(data);
    this->head = buildnode(data, labels);
}

int *DecisionTree::predict(InputData data){
    TreeNode *node = this->head;
    while(node->type != 2){
        TreeBranch *b = static_cast<TreeBranch*>(node);
        if(data[b->col] >= b->value)
            node = b->right;
        else
            node = b->left;
    }
    return static_cast<TreeLeaf*>(node)->freqs;
}

void DecisionTree::calcUniqValues(const std::vector<InputData> &data){
    this->uvalues = new std::unordered_set<InputValue>[data[0].size()];
    for(int i = 0; i < data[0].size(); i++){
        for(int j = 0; j < data.size(); j++){
            uvalues[i].insert(data[j][i]);
        }
    }
    /*for(int i = 0; i < data[0].size(); i++){
        for(auto val : uvalues[i]){
            printf("%d\n", val);
        }
    }*/
    //std::vector<InputValue>* res = new std::vector[this->input_length];
}

void DecisionTree::divideSet(const std::vector<InputData> &data, const std::vector<OutputData> &labels,
        int col, InputValue value, 
        std::vector<InputData> *s1, std::vector<InputData> *s2,
        std::vector<OutputData> *l1, std::vector<OutputData> *l2){
    for(int i = 0; i < data.size(); i++){
        if (data[i][col] >= value){
            s1->push_back(data[i]);
            l1->push_back(labels[i]);
        }
        else{
            s2->push_back(data[i]);
            l2->push_back(labels[i]);
        }
    }
}

TreeNode *DecisionTree::buildnode(const std::vector<InputData> &data, const  std::vector<OutputData> &labels){
    double current_score = this->ginii(labels);
    printf("score %f %d\n", current_score, labels.size());
    double best_gain = 0.0;
    std::vector<InputData>  ms1, ms2;
    std::vector<OutputData> ml1, ml2;
    InputValue best_value;
    int best_col = -1;

    for(int col = 0; col < data[0].size(); col++){
        for(auto &val : this->uvalues[col]){
            std::vector<InputData> s1, s2;
            std::vector<OutputData> l1, l2;
            this->divideSet(data, labels, col, val, &s1, &s2, &l1, &l2);
            double p = (double(s1.size()))/data.size();
            double gain = current_score - p*this->ginii(l1) - (1 - p) * this->ginii(l2);
            //printf("gain %f %d %d %d %d\n", gain, s1.size(), s2.size(), col, val);
            if (gain > best_gain &&
                    l1.size() > 0 && l2.size() > 0){
                best_value = val;
                best_gain = gain;
                best_col = col;
                ml1 = l1; ml2 = l2; ms1 = s1; ms2 = s2;
            }
        }
    }
    if (best_gain > 0){
        TreeBranch *res = new TreeBranch();
        res->left  = buildnode(ms2, ml2);
        res->right = buildnode(ms1, ml1);
        res->col = best_col;
        res->value = best_value;
        return res;
    }
    TreeLeaf *res = new TreeLeaf();
    res->freqs = this->getFreq(labels);
    res->len = this->num_of_classes;
    return res;
}

double DecisionTree::ginii(const std::vector<OutputData> &labels){
    int* freqs = this->getFreq(labels);
    double imp = 0;
    for(int i = 0; i < this->num_of_classes; i++){
        double p1 = freqs[i]/(double)labels.size();
        for(int j = 0; j < this->num_of_classes; j++){
            if (i == j) continue;
            double p2 = freqs[j]/(double)labels.size();
            imp += p1*p2;
        }
    }
    delete [] freqs;
    return imp;
}

int DecisionTree::getNumOfClasses(std::vector<OutputData> labels){
    OutputData max = labels[0];
    for(int i = 1; i < labels.size(); i++){
        if (labels[i] > max) max = labels[i];
    }
    return max + 1;
}

int * DecisionTree::getFreq(std::vector<OutputData> labels){
    int *res = new int[num_of_classes];
    memset(res, 0, sizeof(OutputData) * num_of_classes);
    for(int i = 0; i < labels.size(); i++){
        res[labels[i]]++;
    }
    return res;
}


DecisionTree::DecisionTree(){
}
