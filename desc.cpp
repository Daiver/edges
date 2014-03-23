#include "desc.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

void patchesToVec(cv::Mat img_o, std::vector<float> *res){
    cv::Mat img;
    cv::pyrDown(img_o, img);
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
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    cv::phase(Sx, Sy, ori, true);
    cv::Mat f1 = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat f2 = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat f3 = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat f4 = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            float p = mag.at<float>(i, j);
            float f = ori.at<float>(i, j);
            //float f = cv::fastAtan2(gradY.at<short>(i, j), gradX.at<short>(i, j));
            //printf("%d %d %f\n", i, j, f);
            if(p > 1.0){
                if(f < 90.0)
                    f1.at<float>(i, j) = p;
                else if(f >= 90.0 && f < 180.0)
                    f2.at<float>(i, j) = p;
                else if(f >= 180.0 && f < 270.0)
                    f3.at<float>(i, j) = p;
                else if(f >= 270.0 && f < 360.0)
                    f4.at<float>(i, j) = p;
            }
            //res->push_back(f);
        }
    }
    cv::Mat F[] = {f1,f2,f3,f4};
    for(int k = 0; k < 4;  k++){
        cv::normalize(F[k], F[k], 0, 255, cv::NORM_MINMAX);
        for(int i = 0; i < img.rows; i++){
            for(int j = 0; j < img.cols; j++){
                float p = F[k].at<float>(i, j);
                res->push_back(p);
            }
        }
    }
}

