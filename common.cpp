#include "common.h"

#include <iostream>

void read_imgList(const std::string& filename, std::vector<cv::Mat>* images) {
    std::ifstream file(filename.c_str(), std::ifstream::in);
    std::string line;
    while (std::getline(file, line)) {
        images->push_back(cv::imread(line, 0));
    }
}

void read_imgList2(const std::string& filename, std::vector<cv::Mat>* images, 
        std::vector<cv::Mat>* groundTruth){
    std::ifstream file(filename.c_str(), std::ifstream::in);

    std::string line, img_dir_name, gT_dir_name;
    std::getline(file, img_dir_name);
    std::getline(file, gT_dir_name);
    std::cout << img_dir_name << std::endl << gT_dir_name << std::endl;
    while (std::getline(file, line)) {
        images->push_back(cv::imread(img_dir_name + line + ".jpg", 0));
        groundTruth->push_back(cv::imread(gT_dir_name + line + ".mat_1.png", 0));
    }

}

void cutPatchesFromImage(cv::Mat img, std::vector<cv::Mat>* patches){
    int w = 16;
    for (int i = 0; i < img.rows; i+=20){
        for (int j = 0; j < img.cols; j+=20){
            cv::Mat tileCopy = img(cv::Range(i, std::min(i + w, img.rows)),
                 cv::Range(j, std::min(j + w, img.cols))).clone();
            if (tileCopy.rows != w || tileCopy.cols != w) continue;
            patches->push_back(tileCopy);
        }
    }
}
