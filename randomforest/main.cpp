#include <stdio.h>
#include <vector>
#include <iostream>
#include "decisiontree.h"
#include "randomforest.h"
#include "common.h"


void test1(){
    DecisionTree tree;
    std::vector<std::vector<int> > data ={
        {1,2,3},
        {4,1,9},
        {0,1,1},
        {3,6,3},
        {0,0,0},
        {5,5,5},
        {2,3,8},
    };
    std::vector<int> labels = {
        0, 1, 0, 2, 0, 1, 2
    };
    tree.train(data, labels);
    printf("BEFORE SHOW\n");
    tree.head->show();
    printf("AFTER SHOW\n");
    InputData sample = {4, 0, 0};
    printV(tree.predict(sample), tree.num_of_classes);
    //tree.num_of_classes = tree.getNumOfClasses(labels);
    //int *freqs = tree.getFreq(labels);
    //printf("%ld\n", tree.num_of_classes);
    //printV(freqs, tree.num_of_classes);
    //printf("%f\n", tree.ginii(labels));
    //std::vector<InputData> s1, s2;
    //tree.divideSet(data, 1, 5, &s1, &s2);
    //for(auto &x : s1) printV(x, x.size());
    //printf("-----\n");
    //for(auto &x : s2) printV(x, x.size());
}

int main() {
    test1();
    printf("Start\n");
    return 0;
}
