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
    {
        cv::Mat img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
        cv::cvtColor(img, img, CV_BGR2Luv);
        cv::imshow("1", img);
        img.convertTo(img, CV_32FC3);
        cv::Mat M, O;
        gradientMagnitude(img, M, O);
        //cv::normalize(M, M, 0, 1.0, cv::NORM_MINMAX);
        cv::imshow("2", M);
        cv::imshow("3", O);
        std::vector<cv::Mat> ori;
        gradientHist(M, O, 2, &ori);
        for(int i = 0; i < ori.size(); i++){
            char name[128];
            sprintf(name, "o-%d", i);
            cv::normalize(ori[i], ori[i], 0, 1.0, cv::NORM_MINMAX);
            cv::imshow(name, ori[i]);
        }
    }

    cv::waitKey();
    {
        cv::Mat img2 = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
        cv::cvtColor(img2, img2, CV_BGR2Luv);
        cv::imshow("1n", img2);
        img2.convertTo(img2, CV_32FC3);
        cv::Mat M2, O2;
        gradientMagnitude(img2, M2, O2);
        //cv::normalize(M, M, 0, 1.0, cv::NORM_MINMAX);
        cv::imshow("2n", M2);
        cv::imshow("3n", O2);

        std::vector<cv::Mat> ori2;
        gradientHist(M2, O2, 2, &ori2);
        for(int i = 0; i < ori2.size(); i++){
            char name[128];
            sprintf(name, "on-%d", i);
            cv::normalize(ori2[i], ori2[i], 0, 1.0, cv::NORM_MINMAX);
            cv::imshow(name, ori2[i]);
        }

        cv::waitKey();
    }
        cv::waitKey();
    //cv::pyrDown(img, img);
    /*gradientMagnitude(img, M, O);
    cv::imshow("2n", M);
    cv::imshow("3n", O);
    ori.clear();
    gradientHist(M, O, 2, &ori);
    for(int i = 0; i < ori.size(); i++){
        char name[128];
        sprintf(name, "o-%d", i);
        cv::normalize(ori[i], ori[i], 0, 1.0, cv::NORM_MINMAX);
        cv::imshow(name, ori[i]);
    }
    cv::waitKey();

    gradientMagnitude(img, M, O);
    cv::imshow("2m", M);
    cv::imshow("3m", O);
    ori.clear();
    gradientHist(M, O, 2, &ori);
    for(int i = 0; i < ori.size(); i++){
        char name[128];
        sprintf(name, "o-%d", i);
        cv::normalize(ori[i], ori[i], 0, 1.0, cv::NORM_MINMAX);
        cv::imshow(name, ori[i]);
    }
    cv::waitKey();
    */


    cv::Mat img2;
    /*cv::pyrDown(img, img2);
    {
        cv::Mat M2, O2;
        gradientMagnitude(img2, M2, O2);
        std::vector<cv::Mat> ori;
        gradientHist(M2, O2, 2, &ori);
        for(int i = 0; i < ori.size(); i++){
            char name[128];
            sprintf(name, "o-%d", i);
            cv::normalize(ori[i], ori[i], 0, 1.0, cv::NORM_MINMAX);
            cv::imshow(name, ori[i]);
        }
        cv::normalize(M2, M2, 0, 1.0, cv::NORM_MINMAX);
        cv::imshow("2", M2);
        cv::normalize(O2, O2, 0, 1.0, cv::NORM_MINMAX);
        cv::imshow("3", O2);
        cv::waitKey();
    }*/
//img.put(0,0,buf1);
    printf("End\n");
}
