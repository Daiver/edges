#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

void read_imgList(const std::string& filename, std::vector<cv::Mat>* images) {
    std::ifstream file(filename.c_str(), std::ifstream::in);
    std::string line;
    while (std::getline(file, line)) {
        images->push_back(cv::imread(line, 0));
    }
}

void cutPatchesFromImage(cv::Mat img, std::vector<cv::Mat>* patches){
    int w = 16;
    for (int i = 0; i < img.rows; i+=100){
        for (int j = 0; j < img.cols; j+=100){
            cv::Mat tileCopy = img(cv::Range(i, std::min(i + w, img.rows)),
                 cv::Range(j, std::min(j + w, img.cols))).clone();
            if (tileCopy.rows != w || tileCopy.cols != w) continue;
                patches->push_back(tileCopy);
        }
    }
}

int main(){
    std::vector<cv::Mat> images;
    read_imgList("images.txt", &images);
    for(auto& img : images){
        cv::imshow("", img);
        for(int i = 0 ; i < img.rows; i++){
            for(int j = 0 ; j < img.rows; j++){
                printf("%d ", img.at<uchar>(i, j));
            }
            printf("\n");
        }
        std::vector<cv::Mat> patches;
        cutPatchesFromImage(img, &patches);
        for(auto &tileCopy : patches){
            cv::imshow("1", tileCopy);
            cv::waitKey();
        }
    }
    return 0;
}
