#include "gradientMex.cpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>

float* extractRawData(cv::Mat &img){
    std::vector<cv::Mat> channels;
    cv::split(img, channels);   
    float *res = new float[img.channels() * img.rows * img.cols];
    cv::Size orig_size = img.size();
    std::cout<<">"<<img.channels()<<std::endl;
    int fin_size = orig_size.height * orig_size.width;

    for(int i = 0; i < channels.size(); i++){
        float *buf = (float*)channels[i].data;
        memcpy(res + fin_size*i, buf, sizeof(int) * fin_size);
    }
    return res;
}

int main(){
    cv::Mat img = cv::imread("imgs/img/1.jpg");
    img.convertTo(img, CV_32FC3);
    cv::Size orig_size = img.size();
    cv::Mat M = cv::Mat::zeros(orig_size, CV_32FC1);
    cv::Mat O = cv::Mat::zeros(orig_size, CV_32FC1);
    cv::Mat H = cv::Mat::zeros(orig_size, CV_32FC1);
    float *img_buf = extractRawData(img);
    //(float*)img.data;//new float[orig_size.height * orig_size.width];
    float *M_buf   = (float*)M.data;
    float *O_buf   = (float*)O.data;
    float *H_buf   = (float*)O.data;
    //for(int i = 0; i < orig_size.width * orig_size.height/2; i++){
    //    img_buf[i*3] = 0.2;
    //}
    cv::imshow("1", img);
    gradMag(img_buf, M_buf, O_buf, orig_size.height, orig_size.width,3,0);
    /*for(int i = 0; i < orig_size.width * orig_size.height/2; i++){
        M_buf[i*3] = 0.2;
    }*/
    cv::imshow("2", M);
    //cv::normalize(O, O, 0, .5, cv::NORM_MINMAX);
    //std::cout << O;
    cv::imshow("3", O);
    cv::waitKey();
    //img.put(0,0,buf1);
    printf("End\n");
}
