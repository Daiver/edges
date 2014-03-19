#include "desc.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

void patchesToVec(cv::Mat img, std::vector<float> *res){
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            cv::Vec3b p = img.at<cv::Vec3b>(i, j);
            res->push_back(p[0]);
            res->push_back(p[1]);
            res->push_back(p[2]);
        }
    }
    cv::Mat gray;
    cv::cvtColor(img, gray, CV_Lab2BGR);
    cv::cvtColor(gray, gray, CV_BGR2GRAY);
    /*cv::Mat gradY;
    cv::Sobel(gray, gradY, CV_16S, 0, 1);
    //cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX);
    cv::Mat gradX;
    cv::Sobel(gray, gradX, CV_16S, 1, 0);
    cv::Mat gradF;
    cv::Sobel(gray, gradF, CV_16S, 1, 1);
    cv::normalize(gradF, gradF, 0, 255, cv::NORM_MINMAX);*/
    cv::Mat Sx;
    cv::Sobel(img, Sx, CV_32F, 1, 0, 3);
    cv::Mat Sy;
    cv::Sobel(img, Sy, CV_32F, 0, 1, 3);

    cv::Mat mag, ori;
    cv::magnitude(Sx, Sy, mag);
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
    cv::phase(Sx, Sy, ori, true);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            float p = mag.at<float>(i, j);
            float f = ori.at<float>(i, j);
            //float f = cv::fastAtan2(gradY.at<short>(i, j), gradX.at<short>(i, j));
            //printf("%d %d %f\n", i, j, f);
            res->push_back(p);
            res->push_back(f);
        }
    }
}

