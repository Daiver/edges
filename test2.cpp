
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
    for (int i = 0; i < img.rows; i+=8){
        for (int j = 0; j < img.cols; j+=8){
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
    std::vector<std::vector<float>> tmp_data(img_patches.size());
    for(int i = 0; i < tmp_data.size(); i++){
        patchesToVec(img_patches[i], &tmp_data[i]);
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
    std::vector<cv::Mat> gt_patches2;
    std::vector<std::vector<float>> data;

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

    gt_patches2 = gt_patches;
    data = tmp_data;

    srand(NULL);

    printf("dataset size: %d\n", data.size());
    printf("features len: %d\n", data[0].size());
    RandomForest tree(8);
    tree.train(data, gt_patches2);
    cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
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
