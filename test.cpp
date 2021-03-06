#include "common.h"

#include "decisiontree.h"
#include "randomforest.h"
#include "desc.h"
#include "detect.h"
#include "dimSort.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <unistd.h>

void dimSortTest(){
    printf("Start\n");
    std::vector<std::vector<float>> train_data;
    int dt_size = 30000; int ft_size = 7228;
    //int dt_size = 10; int ft_size = 7;
    srand(time(NULL));
    for(int i = 0; i < dt_size; i++){
        std::vector<float> tmp;
        for(int j = 0; j < ft_size; j++){
            tmp.push_back(rand() % 1000);
        }
        train_data.push_back(tmp);
    }
    std::vector<int> f_idxs;
    for(int i = 0; i < ft_size/5; i++){
        f_idxs.push_back(rand() % ft_size);
    }
    std::vector<int> data_idx;
    for(int i = 0; i < dt_size/3; i++){
        data_idx.push_back(rand() % dt_size);
    }
    /*
    for(auto x:train_data){
        for(auto y:x){
            printf("%f ", y);
        }
        printf("\n");
    }*/
    int **idxs = dimSort(&train_data, f_idxs, &data_idx);
    //idxs[0][0] = 0;
    for(int fid = 0; fid < f_idxs.size(); fid++){
        for(int i = 1; i < data_idx.size(); i++){
            float val0 = train_data[data_idx[idxs[fid][i - 1]]][f_idxs[fid]];
            float val1 = train_data[data_idx[idxs[fid][i]]][f_idxs[fid]];
            //printf("%d %d %f %f\n", f_idxs[fid], data_idx[idxs[fid][i]], val0, val1);
            if(val1 < val0){
                printf("ERRR \n");
                printf("%d %d %f %f\n", f_idxs[fid], data_idx[idxs[fid][i]], val0, val1);
            }
        }
        //printf("\n");
    }
    printf("End\n");
}

void dimSortTest2(){
    for(int i = 0; i < 10; i++) dimSortTest();
}

void chnTest(){
     cv::Mat img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
    //cv::Mat img = cv::imread("imgs/img/1.jpg");
    //img.convertTo(img, CV_32FC3);
    cv::cvtColor(img, img, CV_BGR2Luv);

    cv::imshow("orig ", img);
    std::vector<cv::Mat> chnReg, chnSim;
    imageChns(img, &chnReg, &chnSim);
    printf("%d %d \n", chnReg.size(), chnSim.size());
    //cv::imwrite("tmp.bmp", chnReg[4]);
    for (int i = 0; i < chnReg.size();i++){
        char name[100];
        sprintf(name, "dump/r-%d.png\n", i);
        printf(name);
        //std::cout << chnReg[i];
        cv::Mat tmp;
        cv::normalize(chnReg[i], tmp, 0, 255.0, cv::NORM_MINMAX);
        cv::imwrite(name, tmp);

        cv::imshow(name, tmp);
        cv::normalize(chnSim[i], tmp, 0, 255.0, cv::NORM_MINMAX);
        sprintf(name, "dump/s-%d.png\n", i);
        cv::imshow(name, tmp);
        cv::imwrite(name, tmp);
        //cv::waitKey();
    }
    cv::waitKey();
}

void patchesTest(){
    cv::Mat img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
    cv::Mat gt = cv::imread("/home/daiver/BSR/BSDS500/data/groundImages/train/100075.mat_1.png", 0);
    //cv::Mat img = cv::imread("imgs/img/1.jpg");
    //cv::cvtColor(img, img, CV_BGR2Luv);
    cv::imshow("orig ", img);
    cv::imshow("gt ", gt);
    cv::waitKey(1);
    std::vector<cv::Mat> chnReg, chnSim;
    imageChns(img, &chnReg, &chnSim);
    std::vector<std::vector<float>> descs;
    std::vector<cv::Mat> gt_patches;
    printf("Extracting....\n");
    chnsToVecs(chnReg, chnSim, img, gt, &descs, &gt_patches, 10, 50);
    printf("%d %d\n", descs.size(), gt_patches.size());
}

int main(){
    //dimSortTest2();
    chnTest();
    //patchesTest();
    return 0;
}
