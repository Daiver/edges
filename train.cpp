
#include "common.h"

#include "decisiontree.h"
#include "randomforest.h"
#include "desc.h"
#include "detect.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <unistd.h>


int main(){
    std::vector<cv::Mat> images, gtruth;
    read_imgList2("images5.txt", &images, &gtruth);
    /*for(int i = 0; i < images.size(); i++){
        //cv::imshow("image", images[i]);
        cv::Mat tmp;
        cv::normalize(gtruth[i], tmp, 0, 255, cv::NORM_MINMAX);
        //cv::imshow("edges", tmp);
        //cv::waitKey();
    }*/
    std::vector<cv::Mat> img_patches, gt_patches;
    for(int i = 0; i < images.size(); i++){
        cutPatchesFromImage3(images[i], gtruth[i], &img_patches, &gt_patches, 2000, 2000);
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
    //std::vector<cv::Mat> gt_patches2;
    std::vector<std::vector<float>> data(img_patches.size());
    //std::vector<std::vector<float>> tmp_data(img_patches.size());
    for(int i = 0; i < data.size(); i++){
        patchesToVec(img_patches[i], &data[i]);
        //patchesToVec(img_patches[i], &tmp_data[i]);
    }
    /*cv::Mat to_pca(tmp_data.size(), tmp_data[0].size(), CV_32F);
    for(int i = 0; i < tmp_data.size(); i++){
        for(int j = 0; j < tmp_data[0].size(); j++){
            to_pca.at<float>(i, j) = tmp_data[i][j];
        }
    }*/
    //std::vector<std::vector<float>> data(img_patches.size());
    /*printf("Start pca\n");
    cv::PCA pca(to_pca, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.82);
    printf("End pca # %d\n", pca.eigenvectors.rows);
    for(int i = 0; i < tmp_data.size(); i++){
        cv::Mat tmp_mat = pca.project(to_pca.row(i));
        for(int j = 0; j < tmp_data[0].size(); j++){
            data[i].push_back(tmp_mat.at<float>(0, j));
        }
    }*/

    /*int neg_size = 0;
    for(int i = 0; i < tmp_data.size(); i++){
        //double std = cv::std(gt_patches[i])[0];
        cv::Scalar mean, std;
        cv::meanStdDev(gt_patches[i], mean, std);
        //printf("%f\n", std[0]);
        if(std[0] == 0.0) neg_size++;
        if(neg_size < 400 || std[0] != 0.0){
            data.push_back(tmp_data[i]);
            gt_patches2.push_back(gt_patches[i]);
        }
    }*/

    //gt_patches2 = gt_patches;
    //data = tmp_data;
    
    printf("dataset size: %d\n", data.size());
    printf("features len: %d\n", data[0].size());
    //printf("After reading desc\n");
    //sleep(10);

    srand(NULL);

    RandomForest tree(8);
    tree.train(data, gt_patches);
    tree.save("../model/forest");
    for(int i = 0; i < 8;i++){
        //tree.ansamble[i].head->show();
    }
    cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
    cv::Mat test_res = detect(tree, test_img);
    cv::imwrite("test1.jpg", test_res);
    test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/test/29030.jpg");
    test_res = detect(tree, test_img);
    cv::imwrite("test2.jpg", test_res);
    return 0;

    //cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
    //cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/test/29030.jpg");
    //cv::Mat test_res = reproduce(tree, test_img);
    //cv::imshow("ORIG", test_img);
    //cv::imshow("rep", test_res);
    //cv::waitKey();
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
        std::vector<float> desc;
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
        auto res = ress[ress.size() - 1];
        cv::normalize(res, tmp2, 0, 255, cv::NORM_MINMAX);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::imshow("F", tmp2);
        cv::waitKey();
    }

    return 0;
}
