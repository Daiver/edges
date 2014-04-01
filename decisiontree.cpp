#include "defines.h"
#include "decisiontree.h"

#include <stdio.h>
#include <fstream>
#include <string.h>
#include <vector>
#include <math.h>
//#include <unordered_set>
#include <set>
#include <opencv2/core/core.hpp>

void DecisionTree::train(
        const std::vector<InputData> *data, 
        std::vector<int> &data_idx,
        std::vector<cv::Mat> segments){
    //this->num_of_classes = getNumOfClasses(labels);
    this->calcUniqValues(data);
    this->train_data = data;
    //std::vector<int> data_idx;
    /*for(int i = 0; i < data->size(); i++)
        data_idx.push_back(i);*/
    this->head = buildnode(data_idx, segments, 0);
}

cv::Mat DecisionTree::predict(InputData data){
    TreeNode *node = this->head;
    while(node->type != 2){
        TreeBranch *b = static_cast<TreeBranch*>(node);
        if(data[b->col] >= b->value)
            node = b->right;
        else
            node = b->left;
    }
    return static_cast<TreeLeaf*>(node)->patch;
    //return static_cast<TreeLeaf*>(node)->freqs;
}

void DecisionTree::calcUniqValues(const std::vector<InputData> *data){
    this->uvalues = new std::set<InputValue>[data->at(0).size()];
    for(int i = 0; i < data->at(0).size(); i++){
        for(int j = 0; j < data->size(); j++){
            uvalues[i].insert(data->at(j)[i]);
        }
    }
    for(int i = 0; i < data->at(0).size(); i++){
        if(uvalues[i].size() > 5000)
            printf("USIZE %d\n", uvalues[i].size());
    }
    /*for(int i = 0; i < data[0].size(); i++){
        for(auto val : uvalues[i]){
            printf("%d\n", val);
        }
    }*/
    //std::vector<InputValue>* res = new std::vector[this->input_length];
}

void DecisionTree::divideSet(
        //const std::vector<InputData> &data, 
        const std::vector<int> &data_idx, 
        const std::vector<OutputData> &labels,
        //const std::vector<cv::Mat> &seg,
        int col, InputValue value, 
        std::vector<OutputData> *l1, std::vector<OutputData> *l2,
        //std::vector<cv::Mat> *g1, std::vector<cv::Mat> *g2,
        std::vector<int> *i1, std::vector<int> *i2){
    for(int i = 0; i < data_idx.size(); i++){
        if ((this->train_data->at(data_idx[i]))[col] >= value){
            i1->push_back(data_idx[i]);
            //s1->push_back(data[i]);
            l1->push_back(labels[i]);
            //g1->push_back(seg[i]);
        }
        else{
            i2->push_back(data_idx[i]);
            //s2->push_back(data[i]);
            l2->push_back(labels[i]);
            //g2->push_back(seg[i]);
        }
    }
}

void DecisionTree::finalDivide(
        const std::vector<int> &data_idx, 
        const std::vector<OutputData> &labels,
        //const std::vector<cv::Mat> &seg,
        int col, InputValue value, 
        std::vector<OutputData> *l1, std::vector<OutputData> *l2,
        //std::vector<cv::Mat> *g1, std::vector<cv::Mat> *g2,
        std::vector<int> *i1, std::vector<int> *i2){
    for(int i = 0; i < data_idx.size(); i++){
        if ((this->train_data->at(data_idx[i]))[col] >= value){
            i1->push_back(data_idx[i]);
            //s1->push_back(data[i]);
            l1->push_back(i);
            //g1->push_back(seg[i]);
        }
        else{
            i2->push_back(data_idx[i]);
            //s2->push_back(data[i]);
            l2->push_back(i);
            //g2->push_back(seg[i]);
        }
    }
}


TreeNode *DecisionTree::buildnode(
        //const std::vector<InputData> &data, 
        const std::vector<int> &data_idx, 
        const std::vector<cv::Mat> &segments, int depth){
    /*for(int i = 0; i < segments.size(); i++){
        char name[100];
        sprintf(name, "seg %d", i % 10);
        printf("%d\n", i);
        cv::Mat tmp;
        cv::normalize(segments[i], tmp, 0, 255, cv::NORM_MINMAX);
        cv::pyrUp(tmp, tmp);
        cv::pyrUp(tmp, tmp);
        cv::imshow(name, tmp);
        if(i%10 == 0) cv::waitKey();
    }*/

    int num_of_classes, seg_idx;
    std::vector<int> labels(segments.size(), 0);
    selectFeaturesFromPatches(segments, &labels, &num_of_classes, &seg_idx);

    double current_score = this->ginii(labels, num_of_classes);
#ifdef DECISION_TREE_DEBUG
    printf("score %f %d depth %d\n", current_score, labels.size(), depth);
#endif
    double best_gain = 0.0;
    //std::vector<InputData>  ms1, ms2;
    std::vector<int> ml1, ml2;
    //std::vector<cv::Mat> ml1, ml2;
    InputValue best_value;
    int best_col = -1;

    //for(int col = 0; col < this->train_data[0].size(); col++){
    int m_small = (int)sqrt(this->train_data[0].size());
    for(int col_idx = 0; col_idx < m_small; col_idx++){
        int col = (int)rand() % this->train_data->at(0).size();
#ifdef DECISION_TREE_DEBUG
        if(col_idx % 500 == 0)
            printf("col %d %d\n", col_idx, this->uvalues[col].size());
#endif
        for(auto &val : this->uvalues[col]){
            std::vector<int> i1, i2;
            std::vector<OutputData> l1, l2;
            //std::vector<cv::Mat>    g1, g2;
            this->divideSet(data_idx, labels, 
                    //segments, 
                    col, val, 
                    &l1, &l2, //&g1, &g2, 
                    &i1, &i2);
            double p = ((double)(l1.size()))/data_idx.size();
            double gain = current_score - 
                p*this->ginii(l1, num_of_classes) - 
                (1 - p) * this->ginii(l2, num_of_classes);
            //printf("gain %f %d %d %d %d\n", gain, s1.size(), s2.size(), col, val);
            if (gain > best_gain &&
                    l1.size() > 0 && l2.size() > 0){
                best_value = val;
                best_gain = gain;
                best_col = col;
                ml1.clear();
                ml2.clear();
                ml1 = i1; ml2 = i2;// ms1 = s1; ms2 = s2;
                //ml1 = g1; ml2 = g2; ms1 = s1; ms2 = s2;
                //ml1 = l1; ml2 = l2; ms1 = s1; ms2 = s2;
            }
            i1.clear(); i2.clear();
        }
    }
    if (best_gain > 0 && depth < 84){
        TreeBranch *res = new TreeBranch();
        std::vector<cv::Mat> g1, g2;
        std::vector<OutputData> l1, l2; 
        ml1.clear(); ml2.clear();
        //this->divideSet(data_idx, labels, 
        finalDivide(data_idx, labels, 
                //segments, 
                best_col, best_value, 
                &l1, &l2, //&g1, &g2, 
                &ml1, &ml2);
        //std::vector<InputData> s1, s2;
        char name[100];
        cv::destroyAllWindows();
        for(int i = 0; i < l1.size(); i++){
            g1.push_back(segments[l1[i]]);
#ifdef NODE_SHOW_DEBUG
            if(i < 12){
                sprintf(name, "g1 %i", i);
                cv::Mat tmp;
                cv::pyrUp(g1[i], tmp);
                cv::pyrUp(tmp, tmp);
                cv::pyrUp(tmp, tmp);
                cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX);
                printf("g1 %d %d %d l(%d) %f %f %d\n", i, ml1[i], 
                        best_col, best_value, l1[i],
                        this->train_data->at(ml1[i])[best_col], 
                        this->train_data->at(ml1[i])[best_col] >= best_value);
                cv::imshow(name, tmp);
            }
#endif                
            //s1.push_back(data[ml1[i]]);
        }
        for(int i = 0; i < l2.size(); i++){
            g2.push_back(segments[l2[i]]);
#ifdef NODE_SHOW_DEBUG
            if(i < 12){
                sprintf(name, "g2 %i", i);
                cv::Mat tmp;
                cv::pyrUp(g2[i], tmp);
                cv::pyrUp(tmp, tmp);
                cv::pyrUp(tmp, tmp);
                cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX);
                printf("g2 %d %d %d l(%d) %f %f %d\n", i, ml2[i], 
                        best_col, best_value, labels[l2[i]],
                        this->train_data->at(ml2[i])[best_col], 
                        this->train_data->at(ml2[i])[best_col] < best_value);
                cv::imshow(name, tmp);
            }
#endif
            //s2.push_back(data[ml2[i]]);
        }
#ifdef NODE_SHOW_DEBUG
        cv::waitKey();
#endif

        res->left  = buildnode(ml2, g2, depth + 1);
        //res->left  = buildnode(ms2, ml2);
        res->right = buildnode(ml1, g1, depth + 1);
        //res->right = buildnode(ms1, ml1);
        res->col = best_col;
        res->value = best_value;
        return res;
    }
    TreeLeaf *res = new TreeLeaf();
    res->freqs = this->getFreq(labels, num_of_classes);
    res->len = num_of_classes;
    res->patch = segments[seg_idx].clone();
    return res;
}

double DecisionTree::ginii(const std::vector<OutputData> &labels, int num_of_classes){
    int* freqs = this->getFreq(labels, num_of_classes);
    double imp = 0;
    for(int i = 0; i < num_of_classes; i++){
        double p1 = freqs[i]/(double)labels.size();
        for(int j = 0; j < num_of_classes; j++){
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

int * DecisionTree::getFreq(std::vector<OutputData> labels, int num_of_classes){
    int *res = new int[num_of_classes];
    memset(res, 0, sizeof(OutputData) * num_of_classes);
    for(int i = 0; i < labels.size(); i++){
        res[labels[i]]++;
    }
    return res;
}


DecisionTree::DecisionTree(){
}

TreeNode* readNodeFromFile(std::ifstream &in){
    int type;
    in >> type;
    if(type == 1){
        std::string s;
        in >> s;
        int col ; float value;
        in >> col >> value;
        TreeBranch *b = new TreeBranch();
        b->col = col;
        b->value = value;
        b->left = readNodeFromFile(in);
        b->right = readNodeFromFile(in);
        return b;
    }
    else{
        std::string s;
        in >> s;
        int rows, cols;
        in >> rows >> cols;
        cv::Mat p(rows, cols, CV_8U);
        for(int i = 0; i < p.rows; i++){
            for(int j = 0; j < p.rows; j++){
                int t;
                in >> t;
                p.at<uchar>(i, j) = t;
            }
        }
        TreeLeaf *res = new TreeLeaf();
        res->len = 0;
        res->patch = p;
        return res;
    }

}

void DecisionTree::load(const char *fname){
    std::ifstream in(fname);
    this->head = readNodeFromFile(in);
    in.close();
}

void writeNodeToFile(std::ofstream &out, TreeNode* node){
    if(node->type == 1){
        TreeBranch *b = static_cast<TreeBranch*>(node);
        out << "1\n" << "TreeBranch\n" << b->col << " " << b->value << "\n";
        writeNodeToFile(out, b->left);
        writeNodeToFile(out, b->right);
    }
    else{
        auto p = static_cast<TreeLeaf*>(node)->patch;
        out << "2\n" << "TreeLeaf\n";
        out << p.rows << " " << p.cols << "\n";
        for(int i = 0; i < p.rows; i++){
            for(int j = 0; j < p.rows; j++){
                out << (int)p.at<uchar>(i, j) << " ";
            }
            out << "\n";
        }
    }
}

void DecisionTree::save(const char *fname){
    std::ofstream out(fname);
    writeNodeToFile(out, this->head);
    out.close();
}


/*
    while(node->type != 2){
        TreeBranch *b = static_cast<TreeBranch*>(node);
        if(data[b->col] >= b->value)
            node = b->right;
        else
            node = b->left;
    }
    return static_cast<TreeLeaf*>(node)->patch;
*/
