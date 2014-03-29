
#include "common.h"

#include "decisiontree.h"
#include "randomforest.h"
#include "desc.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>


cv::Mat reproduce(RandomForest &forest, cv::Mat img_o){
    int img_w = 32;
    int gt_w = 16;
    cv::Mat img;
    cv::cvtColor(img_o, img, CV_BGR2Luv);
    cv::Mat res = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    std::vector<float> desc;
    for (int i = 0; i < img.rows; i+=16){
        for (int j = 0; j < img.cols; j+=16){
            cv::Mat tileCopy = img(
                cv::Range(i, std::min(i + img_w, img.rows)),
                cv::Range(j, std::min(j + img_w, img.cols)));//.clone();
            if (tileCopy.rows == img_w && tileCopy.cols == img_w){
                patchesToVec(tileCopy, &desc);
                std::vector<cv::Mat> ress = forest.predict(desc);
                cv::Mat edges;
                cv::Canny(ress[ress.size() - 1], edges, 0, 1);
                cv::imshow("1", ress[ress.size() - 1]);
                cv::imshow("2", edges);
                //cv::waitKey();
                for(int ii = 0; ii < gt_w; ii++){
                    for(int jj = 0; jj < gt_w; jj++){
                        res.at<uchar>(i + ii, j + jj) = 
                            edges.at<uchar>(ii, jj);
                    }
                }
            }
        }
    }
    cv::Scalar mean, std;
    cv::meanStdDev(res, mean, std);
    printf(">>>%f %f\n", std[0], mean[0]);
    return res;
}

int main(){
    std::vector<cv::Mat> images, gtruth;
    read_imgList2("images2.txt", &images, &gtruth);
    for(int i = 0; i < images.size(); i++){
        //cv::imshow("image", images[i]);
        cv::Mat tmp;
        cv::normalize(gtruth[i], tmp, 0, 255, cv::NORM_MINMAX);
        //cv::imshow("edges", tmp);
        //cv::waitKey();
    }
    std::vector<cv::Mat> img_patches, gt_patches;
    for(int i = 0; i < images.size(); i++){
        cutPatchesFromImage2(images[i], gtruth[i], &img_patches, &gt_patches);
    }
    std::vector<std::vector<float>> tmp_data(img_patches.size());
    for(int i = 0; i < tmp_data.size(); i++){
        patchesToVec(img_patches[i], &tmp_data[i]);
    }
    std::vector<cv::Mat> gt_patches2;
    std::vector<std::vector<float>> data;

    gt_patches2 = gt_patches;
    data = tmp_data;

    srand(NULL);

    printf("dataset size: %d\n", data.size());
    printf("features len: %d\n", data[0].size());
    RandomForest tree(8);
    tree.load("forest");
    for(int i = 0; i < 8;i++){
        tree.ansamble[i].head->show();
    }

    //cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
    cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/test/29030.jpg");
    cv::Mat test_res = reproduce(tree, test_img);
    cv::imshow("ORIG", test_img);
    cv::imshow("rep", test_res);
    cv::waitKey();
    for(int i = 0; i < tmp_data.size(); i++){
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