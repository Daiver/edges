#ifndef __GRADWORK_H__
#define __GRADWORK_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>

//#include "gradientMex.h"

float* extractRawData(cv::Mat &img);
void splitRawArray(float *arr, int rows, int cols, int dims, std::vector<cv::Mat> *res);
void gradientMagnitude(cv::Mat &img, cv::Mat &M, cv::Mat &O);
void gradientHist(cv::Mat &M, cv::Mat &O, int s, std::vector<cv::Mat> *res);

/*

float* extractRawData(cv::Mat &img){
    std::vector<cv::Mat> channels;
    cv::split(img, channels);   
    float *res = new float[img.channels() * img.rows * img.cols];
    cv::Size orig_size = img.size();
    int fin_size = orig_size.height * orig_size.width;

    for(int i = 0; i < channels.size(); i++){
        float *buf = (float*)channels[i].data;
        memcpy(res + fin_size*i, buf, sizeof(float) * fin_size);
    }
    return res;
}

void splitRawArray(float *arr, int rows, int cols, int dims, std::vector<cv::Mat> *res){
    for(int i = 0; i < dims; i++){
        res->push_back(cv::Mat::zeros(rows, cols, CV_32FC1));
        float *buf = (float*)(res->at(i)).data;
        memcpy(buf, arr + (rows*cols)*i, sizeof(float) * (rows*cols));
    }
}

void gradientMagnitude(cv::Mat &img, cv::Mat &M, cv::Mat &O){
    cv::Size orig_size = img.size();
    M = cv::Mat::zeros(orig_size, CV_32F);
    O = cv::Mat::zeros(orig_size, CV_32F);
    float *img_buf = extractRawData(img);
    gradMag(img_buf, (float *)M.data, (float *)O.data, orig_size.width, orig_size.height,img.channels(),1);
    delete[] img_buf;
}

void gradientHist(cv::Mat &M, cv::Mat &O, int s, std::vector<cv::Mat> *res){
    cv::Size orig_size = M.size();
    int h = orig_size.width; int w = orig_size.height;
    const int shrink = 2; //const int s = 2;
    int binSize = std::max(1, shrink/s); //shrink = 2 ; //s {1, 2}
    int nOrients = 4;
    int softBin = 0;
    int hb = h/binSize; int wb = w/binSize;
    int nChns = nOrients;
    float *H_buf = new float[hb * wb * nChns];
    gradHist((float *)M.data, (float *)O.data, H_buf, h, w, binSize, nOrients, softBin, 1);
    splitRawArray(H_buf, wb, hb, nOrients, res);
    delete[] H_buf;
}
*/
#endif
