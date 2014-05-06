//#include "gradWork.cpp"
#include "gradientMex.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>

int main(){
    //cv::Mat img = cv::imread("imgs/img/1.jpg");
    cv::Mat img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
    cv::imshow("1", img);
    cv::cvtColor(img, img, CV_BGR2Luv);
    img.convertTo(img, CV_32FC3);
    cv::Mat M, O;
    gradientMagnitude(img, M, O);
    std::vector<cv::Mat> ori;
    gradientHist(M, O, 2, &ori);
    for(int i = 0; i < ori.size(); i++){
        char name[128];
        sprintf(name, "o-%d", i);
        cv::normalize(ori[i], ori[i], 0, 1.0, cv::NORM_MINMAX);
        cv::imshow(name, ori[i]);
    }
    cv::normalize(M, M, 0, 1.0, cv::NORM_MINMAX);
    cv::imshow("2", M);
    cv::normalize(O, O, 0, 1.0, cv::NORM_MINMAX);
    //std::cout << O;
    cv::imshow("3", O);

    cv::waitKey();
    //img.put(0,0,buf1);
    printf("End\n");
}
