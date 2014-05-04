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
        std::cout<<"<"<<channels[i].channels() << " \n";
        memcpy(res + fin_size*i, buf, sizeof(float) * fin_size);
        cv::Mat tmp = cv::Mat::zeros(orig_size, CV_32F);
        tmp.data = (uchar*)(res + fin_size * i);
        cv::normalize(tmp, tmp, 0, 1.0, cv::NORM_MINMAX);
        cv::imshow("", tmp);
        cv::waitKey();
        //std::copy(buf, buf + fin_size, res + fin_size*i);
    }
    return res;
}

int main(){
    //cv::Mat img = cv::imread("imgs/img/1.jpg");
    cv::Mat img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg", 0);
    cv::imshow("1", img);
    //cv::cvtColor(img, img, CV_BGR2Luv);
    img.convertTo(img, CV_32FC1);
    cv::Size orig_size = img.size();
    cv::Mat M = cv::Mat::zeros(orig_size, CV_32F);
    cv::Mat O = cv::Mat::zeros(orig_size, CV_32F);
    cv::Mat H = cv::Mat::zeros(orig_size, CV_32F);
    //float *img_buf = extractRawData(img);
    float *img_buf = (float*)img.data;//new float[orig_size.height * orig_size.width];
    float *M_buf   = (float*)M.data;
    float *O_buf   = (float*)O.data;
    //float *H_buf   = (float*)O.data;
    //for(int i = 0; i < orig_size.width * orig_size.height/2; i++){
    //    img_buf[i*3] = 0.2;
    //}
    gradMag(img_buf, M_buf, O_buf, orig_size.height, orig_size.width,1,1);
    /*for(int i = 0; i < orig_size.width * orig_size.height/2; i++){
        M_buf[i*3] = 0.2;
    }*/
    cv::normalize(M, M, 0, 1.0, cv::NORM_MINMAX);
    cv::imshow("2", M);
    cv::normalize(O, O, 0, 1.0, cv::NORM_MINMAX);
    //std::cout << O;
    cv::imshow("3", O);
    cv::waitKey();
    //img.put(0,0,buf1);
    printf("End\n");
}
