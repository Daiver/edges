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

/*std::vector<int> getRandIndxs(int size){
    std::vector<int> indxs(size);
    for(int i = 0; i < size; i++) indxs[i] = i;
    for(int i = 0; i < size; i++) swap(&indxs[i], &indxs[rand() % size]);
    return indxs;
}*/

void selectFeaturesFromPatches(std::vector<cv::Mat>& images){
    std::vector<int> indxs;
    for(int i = 0; i < 256; i++){
        indxs.push_back((int)(rand() % (256*256)));
    }
    std::vector<std::vector<int>> zs(images.size());
    for(int i = 0; i < images.size(); i++){
        for(int j = 0; j < 256; j++){
            zs[i].push_back(images[i].data[indxs[j]/256] == images[i].data[indxs[j]%256]);
            printf("%d %d %d\n", zs[i][j], images[i].data[indxs[j]/256], images[i].data[indxs[j]%256]);
        }
    }
    //for(auto &i : indxs){
    //    printf("%d %d %d\n", i, i / 256, i % 256);
    //} 
}

int main(){
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> patches;
    read_imgList("images.txt", &images);
    for(auto& img : images){
        cv::imshow("", img);
        cutPatchesFromImage(img, &patches);
        //for(auto &tileCopy : patches){
            //cv::imshow("1", tileCopy);
            //cv::waitKey();
        //}
    }
    selectFeaturesFromPatches(patches);
    return 0;
}
        /*for(int i = 0 ; i < img.rows; i++){
            for(int j = 0 ; j < img.rows; j++){
                printf("%d ", img.at<uchar>(i, j));
            }
            printf("\n");
        }*/

