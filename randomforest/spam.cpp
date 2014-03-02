#include <stdio.h>
#include "randomforest.h"
#include "common.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>
#include <unordered_set>

std::vector<std::string> split(std::string sentence){
    using namespace std;
    if(sentence.size() == 0) return std::vector<std::string>();
    istringstream iss(sentence);
    vector<string> tokens{istream_iterator<string>{iss},
         istream_iterator<string>{}};
    return tokens;
}

std::unordered_set<std::string> mkFreqSet(std::vector<std::string> s){
    std::unordered_set<std::string> res;
    for(auto x:s) res.insert(x);
    return res;
}
std::unordered_set<std::string> mkFreqSet(std::string s){
    std::unordered_set<std::string> res;
    for(auto &x : split(s)){
        res.insert(x);
    }
    return res;
}

std::vector<int> stringToSample(std::unordered_set<std::string> &set, std::vector<std::string> vec){
    //auto tokens = split(s);
    auto set2 = mkFreqSet(vec);
    std::vector<int> res;
    for(auto &x:set){
        if(set2.find(x) != set2.end())
            res.push_back(1);
        else res.push_back(0);
    }
    return res;
}

int main() {
    std::ifstream f("spamdata");
    std::unordered_set<std::string> freq;
    while (f){
        std::string s;
        std::getline(f, s);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        //printf("%s\n", s.c_str());
        auto tokens = split(s);
        if (tokens.size() == 0) continue;
        //printf("%s\n", tokens[0].c_str());
        tokens.erase(tokens.begin());
        for (auto x : tokens)  freq.insert(x);
    }
    f.close();
    //for (auto x:freq) printf("%s\n", x.c_str());
    f.open("spamdata");
    std::vector<std::vector<int>> data;
    std::vector<int> labels;
    while (f){
        std::string s;
        std::getline(f, s);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        auto tokens = split(s);
        if (tokens.size() == 0) continue;
        if(tokens[0].compare("ham") == 0)
            labels.push_back(1);
        else
            labels.push_back(0);
        tokens.erase(tokens.begin());
        data.push_back(stringToSample(freq, tokens));
    }
    //printV(labels, labels.size());
    std::vector<std::vector<int>> tdata;
    std::vector<int> tlabels;
    for(int i = 0; i < 500; i++){
        tdata.push_back(data[i]);
        tlabels.push_back(labels[i]);
    }
    //RandomForest forest(2000);
    DecisionTree forest;
    forest.train(tdata, tlabels);
    for(int i = 2000; i < 2200; i++){
        printV(forest.predict(data[i]), forest.num_of_classes);
    }
    forest.head->show(0);
    //forest.ansamble[1].head->show(0);
    /*for(auto x:data){
        printV(x, x.size());
    }*/
    return 0;
}
