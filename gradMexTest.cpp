//#include "gradWork.cpp"
#include "gradientMex.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>

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
