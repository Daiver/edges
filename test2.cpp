#include "common.h"

#include "decisiontree.h"
#include "randomforest.h"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

void patchesToVec(cv::Mat img, std::vector<int> *res){
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
    cv::Mat tmp;
    cv::Sobel(gray, tmp, CV_8U, 0, 1);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            uchar p = tmp.at<uchar>(i, j);
            res->push_back(p);
        }
    }
    cv::Sobel(gray, tmp, CV_8U, 1, 0);
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            uchar p = tmp.at<uchar>(i, j);
            res->push_back(p);
        }
    }


}

int main(){
    std::vector<cv::Mat> images, gtruth;
    read_imgList2("images2.txt", &images, &gtruth);
    for(int i = 0; i < images.size(); i++){
        cv::imshow("image", images[i]);
        cv::Mat tmp;
        cv::normalize(gtruth[i], tmp, 0, 255, cv::NORM_MINMAX);
        cv::imshow("edges", tmp);
        //cv::waitKey();
    }
    std::vector<cv::Mat> img_patches, gt_patches;
    for(int i = 0; i < images.size(); i++){
        cutPatchesFromImage2(images[i], gtruth[i], &img_patches, &gt_patches);
    }
    //printf("%d %d\n", img_patches[0].channels(), img_patches[0].depth() == CV_8U);
    /*for(int i = 0; i < img_patches.size(); i++){
        cv::Mat tmp;
        cv::pyrUp(img_patches[i], tmp);
        cv::pyrUp(tmp, tmp);
        //cv::pyrUp(tmp, tmp);
        cv::cvtColor(tmp, tmp, CV_Lab2BGR);
        cv::imshow("1", tmp);
        cv::Mat tmp2;
        cv::normalize(gt_patches[i], tmp2, 0, 255, cv::NORM_MINMAX);
        //cv::pyrUp(gt_patches[i], tmp);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::imshow("o2", tmp2);
        printf("%d %d %f\n", i, img_patches.size(), cv::sum(gt_patches[i])[0]);
        cv::waitKey();
    }*/
    std::vector<std::vector<int>> data(img_patches.size());
    for(int i = 0; i < data.size(); i++){
        patchesToVec(img_patches[i], &data[i]);
    }
    RandomForest tree(10);
    tree.train(data, gt_patches);
    for(int i = 0; i < data.size(); i++){
        cv::Mat tmp;
        cv::pyrUp(img_patches[i], tmp);
        cv::pyrUp(tmp, tmp);
        cv::cvtColor(tmp, tmp, CV_Lab2BGR);
        cv::imshow("1", tmp);
        cv::Mat tmp2;
        cv::normalize(gt_patches[i], tmp2, 0, 255, cv::NORM_MINMAX);
        //cv::pyrUp(gt_patches[i], tmp);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::imshow("o2", tmp2);
        std::vector<int> desc;
        patchesToVec(img_patches[i], &desc);
        std::vector<cv::Mat> ress = tree.predict(desc);
        int j = 0;
        for(auto res : ress){
            j++;
            cv::normalize(res, tmp2, 0, 255, cv::NORM_MINMAX);
            cv::pyrUp(tmp2, tmp2);
            cv::pyrUp(tmp2, tmp2);
            char nm[100];
            sprintf(nm, "res %d", j);
            cv::imshow(nm, tmp2);
        }
        cv::waitKey();
    }

    return 0;
}
