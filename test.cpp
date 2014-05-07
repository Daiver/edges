#include "common.h"

#include "decisiontree.h"
#include "randomforest.h"
#include "desc.h"
#include "detect.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <unistd.h>

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
        sprintf(name, "r %d\n", i);
        printf(name);
        //std::cout << chnReg[i];
        cv::Mat tmp;
        cv::normalize(chnReg[i], tmp, 0, 1.0, cv::NORM_MINMAX);
        cv::imshow(name, tmp);
        //cv::normalize(chnSim[i], tmp, 0, 1.0, cv::NORM_MINMAX);
        sprintf(name, "s %d", i);
        cv::imshow(name, tmp);
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
    chnTest();
    //patchesTest();
    return 0;
}
