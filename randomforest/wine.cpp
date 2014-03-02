
#include <stdio.h>
#include <fstream>
#include <vector>
#include "common.h"
#include "randomforest.h"

void train(RandomForest &forest){
    std::ifstream f("wine_train");
    std::vector<std::vector<int> > data;
    std::vector<int> labels;
    for(int j = 0; j < 178; j++){//while(f){
        int cls;
        f >> cls;
        labels.push_back(cls);
        float a;
        int b;
        std::vector<int> tmp;
        for(int i = 0; i < 13; i++){
            f >> a;
            b = (int)(a*100);
            tmp.push_back(b);
        }
        data.push_back(tmp);
    }
    forest.train(data, labels);
}

int main(){
    RandomForest forest(20);
    train(forest);
    std::ifstream f("wine_test");
    std::vector<std::vector<int> > data;
    std::vector<int> labels;
    for(int j = 0; j < 178; j++){//while(f){
        int cls;
        f >> cls;
        labels.push_back(cls);
        float a;
        int b;
        std::vector<int> tmp;
        for(int i = 0; i < 13; i++){
            f >> a;
            b = (int)(a*100);
            tmp.push_back(b);
        }
        data.push_back(tmp);
    }

    int error = 0;
    for(int i = 0; i < data.size(); i++){
        auto ans = forest.predict(data[i]);
        int max_i = -1;
        float max = -1;
        for(int j = 0; j < forest.num_of_classes; j++){
            if (ans[j] > max){
                max_i = j;
                max = ans[j];
            }
        }
        if (max_i != labels[i]){
            error += 1;
        }
        printf("%d %d ", labels[i], max_i);

        printV(forest.predict(data[i]), forest.num_of_classes);
    }
    printf("err %f\n", (float)error/data.size());
    return 0;
}
