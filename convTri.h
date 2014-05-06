#ifndef __CONVTRI_H__
#define __CONVTRI_H__

#include "defines.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

static cv::Mat convTri(cv::Mat img, float r){
    cv::Mat f1, f2, kernel, res;
    if (r <= 1){
        f1 = cv::Mat::zeros(1, 3, CV_32F);
        float p = 12/r/(r + 2) - 2;
        f1.at<float>(0, 0) = 1;
        f1.at<float>(0, 1) = p;
        f1.at<float>(0, 2) = 1;
        f1 /= (2 + p);
    }
    else {
        f1 = cv::Mat::zeros(1, (int)r*2 + 1, CV_32F);
        int i = 0;
        for(; i < (int)r; i++){
            f1.at<float>(0, i) = i + 1;
        }
        f1.at<float>(0, i) = r + 1;
        i++;
        for(; i < (int)r*2 + 1; i++){
            f1.at<float>(0, i) = 2*r - i + 1;
        }
        f1 /= (r+1)*(r + 1);
        //f=[1:r r +1 r:-1:1]/(r + 1)^2
    }
    cv::transpose(f1, f2);
    kernel = f2*f1;
    /*std::cout<<f1 <<"\n\n";
    std::cout<<f2 <<"\n\n";
    std::cout<<kernel;*/
    cv::filter2D(img, res, -1, kernel);
    return res;
}

#endif
