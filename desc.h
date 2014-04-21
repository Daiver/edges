#ifndef __DESC_H__
#define __DESC_H__

#include "defines.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

void patchesToVec(cv::Mat img, std::vector<float> *res);
cv::Mat convTri(cv::Mat, float);

//template <int T>
//void gradientMag(cv::Mat img, cv::Mat &M, cv::Mat &O, int normRad, float normConst);
//

template <class T>
void gradientMag(cv::Mat img, cv::Mat &M, cv::Mat &O, int normRad, float normConst){
#ifdef GRAD_MAG_DEBUG
    printf("Compute channels\n");
#endif
    int chnls_size = img.channels();//3;//sizeof(chnls)/sizeof(cv::Mat);
    ///cv::Mat chnls[] = {
    //    cv::Mat::zeros(img.rows, img.cols, img.depth()),
    //    cv::Mat::zeros(img.rows, img.cols, img.depth()),
    //    cv::Mat::zeros(img.rows, img.cols, img.depth()),
    //};/
    //printf("CHNSK %d\n", chnls_size);
    cv::Mat *chnls = new cv::Mat[chnls_size];
    for(int i = 0; i < chnls_size; i++) {
        chnls[i] = cv::Mat::zeros(img.rows, img.cols, img.depth());
    }


    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            cv::Vec3b p = img.at<cv::Vec3b>(i, j);
            for(int k = 0; k < chnls_size; k++){
                chnls[k].at<T>(i, j) = p[k];//WARN
                //chnls[k].at<uchar>(i, j) = p[k];//WARN
            }
        }
    }
    cv::Mat Sx[4];
    cv::Mat Sy[4];
    cv::Mat mag[4];
    Sx[3] = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    Sy[3] = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    mag[3] = cv::Mat::zeros(img.rows, img.cols, CV_32F);
#ifdef GRAD_MAG_DEBUG
    printf("Compute mag\n");
#endif
    for(int k = 0; k < chnls_size; k++){
        cv::Sobel(chnls[k], Sx[k], CV_32F, 1, 0);
        cv::Sobel(chnls[k], Sy[k], CV_32F, 0, 1);
        cv::magnitude(Sx[k], Sy[k], mag[k]);
    }
#ifdef GRAD_MAG_DEBUG
    printf("Compute max mag\n");
#endif
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            float max = 0;
            int ind = 0;
            for(int k = 0; k < chnls_size; k++){
                float p = mag[k].at<float>(i,j);
                if (p > max){
                    max = p;
                    ind = k;
                }
            }
            mag[3].at<float>(i, j) = max;
            Sx[3].at<float>(i, j) = Sx[ind].at<float>(i,j);
            Sy[3].at<float>(i, j) = Sy[ind].at<float>(i,j);
        }
    }
#ifdef GRAD_MAG_DEBUG
    printf("Compute norm\n");
#endif
    cv::Mat M1 = mag[3];
    if (normRad == 0) {
        M = M1; 
        return;
    }
    cv::Mat S = convTri(M1, normRad) + normConst;
    cv::divide(M1, S, M);
    cv::divide(Sx[3], S, Sx[3]);
    cv::divide(Sy[3], S, Sy[3]);
#ifdef GRAD_MAG_DEBUG
    printf("Compute ori\n");
#endif
    O = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::phase(Sx[3], Sy[3], O, true);
    /*for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            if (O.at<float>(i, j) < 0) printf("--- %f\n", O.at<float>(i,j));
            if (O.at<float>(i, j) >3.15) printf("++ %f\n", O.at<float>(i,j));
        }
    }*/
    //O = (3.14 + O)/2.;
}

#endif
