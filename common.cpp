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
        images->push_back(cv::imread(img_dir_name + line + ".jpg"));
        cv::cvtColor(images->at(images->size() - 1), images->at(images->size() - 1), CV_BGR2Luv);
        groundTruth->push_back(cv::imread(gT_dir_name + line + ".mat_1.png", 0));
    }

}


void cutPatchesFromImage2(cv::Mat img, cv::Mat gtruth, std::vector<cv::Mat>* img_patches, std::vector<cv::Mat> *gt_patches){
    int gt_w = 16;
    int img_w = 32;
    for (int i = 0; i < img.rows; i+=8){
        for (int j = 0; j < img.cols; j+=8){
    /*for (int i = 50; i < 200; i+=4){
        for (int j = 50; j < 200; j+=4){*/
            cv::Mat tileCopy = img(
                    cv::Range(i, std::min(i + img_w, img.rows)),
                    cv::Range(j, std::min(j + img_w, img.cols)));//.clone();
            if (tileCopy.rows == img_w && tileCopy.cols == img_w){
                //continue;
                int cI = i + img_w/2;
                int cJ = j + img_w/2;
                //printf("continue %d %d\n", i, j);
                //printf("c %d %d\n", cI, cJ);
                //printf("c %d %d %d %d\n", cI - gt_w/2, cJ - gt_w/2, cI + gt_w/2, cJ + gt_w/2);
                cv::Mat gt_tile = gtruth(
                        cv::Range(cI - gt_w/2, cI + gt_w/2),
                        cv::Range(cJ - gt_w/2, cJ + gt_w/2));//.clone();
                if(gt_tile.rows == gt_w && gt_tile.cols == gt_w){
                    img_patches->push_back(tileCopy);
                    //cv::Canny(gt_tile, gt_tile, 1, 2);
                    gt_patches->push_back(gt_tile);
                }
            }
        }
    }
}

void cutPatchesFromImage(cv::Mat img, std::vector<cv::Mat>* patches){
    int w = 16;
    for (int i = 0; i < img.rows; i+=5){
        for (int j = 0; j < img.cols; j+=5){
            cv::Mat tileCopy = img(cv::Range(i, std::min(i + w, img.rows)),
                 cv::Range(j, std::min(j + w, img.cols))).clone();
            if (tileCopy.rows != w || tileCopy.cols != w) continue;
            patches->push_back(tileCopy);
        }
    }
}
