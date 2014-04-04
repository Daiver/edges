
#include "common.h"

#include "decisiontree.h"
#include "randomforest.h"
#include "desc.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

cv::Mat reproduce2(RandomForest &tree, cv::Mat img_o){
    int img_w = 32;
    int gt_w = 16;
    cv::Mat img;
    std::vector<cv::Mat> img_patches, gt_patches;
    cv::cvtColor(img_o, img, CV_BGR2Luv);
    cv::Mat gt = cv::Mat::zeros(img.rows, img.cols, CV_8U);

    cutPatchesFromImage2(img, gt, &img_patches, &gt_patches);
    std::vector<std::vector<float>> tmp_data(img_patches.size());
    for(int i = 0; i < tmp_data.size(); i++){
        patchesToVec(img_patches[i], &tmp_data[i]);
    }
    std::vector<cv::Mat> gt_patches2;
    std::vector<std::vector<float>> data;

    gt_patches2 = gt_patches;
    data = tmp_data;

    printf("dataset size: %d\n", data.size());
    printf("features len: %d\n", data[0].size());
    for(int i = 0; i < 8;i++){
        tree.ansamble[i].head->show();
    }

    cv::Mat fin_edges(img.rows, img.cols, CV_8U);
    for(int i = 0; i < tmp_data.size(); i++){
        std::vector<float> desc;
        patchesToVec(img_patches[i], &desc);
        std::vector<cv::Mat> ress = tree.predict(desc);
        auto res = ress[ress.size() - 1];
        cv::Mat edges, tmp2;
        cv::Canny(res, edges, 0, 1);
        cv::normalize(res, tmp2, 0, 255, cv::NORM_MINMAX);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::imshow("F", tmp2);
        cv::pyrUp(edges, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::imshow("E", tmp2);
        //printf("i %d\n", i);
        int sti = (i /(fin_edges.cols/8))*8;
        int stj = (i %(fin_edges.cols/8))*8;
        for(int ii = 0; ii < 8;ii++){
            for(int  jj = 0; jj < 8;jj++){
                fin_edges.at<uchar>(ii + sti, jj + stj) = edges.at<uchar>(ii, jj);
            }
        }
    }
    cv::imshow("RR", fin_edges);
    fin_edges = convTri(fin_edges, 1);
    cv::waitKey();
    return fin_edges;
}


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
                cv::Mat tmp;
                cv::pyrUp(tileCopy, tmp);
                cv::pyrUp(tmp, tmp);
                cv::imshow("", tmp);
                patchesToVec(tileCopy, &desc);
                std::vector<cv::Mat> ress = forest.predict(desc);
                cv::Mat edges;
                cv::Canny(ress[ress.size() - 1], edges, 0, 1);
                cv::normalize(ress[ress.size() - 1], tmp, 0, 255, cv::NORM_MINMAX);
                cv::pyrUp(tmp, tmp);
                cv::pyrUp(tmp, tmp);
                cv::pyrUp(tmp, tmp);
                cv::imshow("1", tmp);

                cv::Mat tmp2;
                cv::pyrUp(edges, tmp2);
                cv::pyrUp(tmp2, tmp2);
                cv::pyrUp(tmp2, tmp2);
                cv::imshow("2", tmp2);
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

void convTriTest(){
    cv::Mat test_img2 = cv::imread("/home/daiver/BSR/BSDS500/data/images/test/100099.jpg", 0);

    cv::Mat tmp_img = convTri(test_img2, 8);
    cv::imshow("1", test_img2);
    cv::imshow("2", tmp_img);
    cv::Mat orig_tmp = cv::imread("/home/daiver/tmp.bmp", 0);
    cv::imshow("orig", orig_tmp);
    auto diff = abs(orig_tmp - tmp_img);
    cv::imshow("diff", diff);
    cv::Scalar mean, std;
    cv::meanStdDev(diff, mean, std);
    printf(">>>%f %f\n", std[0], mean[0]);

    cv::waitKey();
    return;
}

void gradMagTest(){
    cv::Mat test_img2 = cv::imread(
            "/home/daiver/BSR/BSDS500/data/images/test/100099.jpg");
    
    cv::Mat mag, ori;
    gradientMag(test_img2, mag, ori, 4, 0.01);
    cv::imshow("1", test_img2);
    printf("show mag %d %d\n", mag.rows, mag.cols);
    cv::imshow("2", mag);
    printf("show ori %d %d\n", ori.rows, ori.cols);
    cv::normalize(ori, ori, 0, 255, cv::NORM_MINMAX);
    cv::imshow("3", ori);
    cv::waitKey();
}
int main(){
    //convTriTest(); return 0;
    //gradMagTest(); return 0;

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

    cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
    //cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/test/29030.jpg");
    cv::Mat test_res = reproduce2(tree, test_img);
    cv::imshow("ORIG", test_img);
    cv::imshow("rep", test_res);
    cv::waitKey();
    cv::Mat fin_edges(test_img.rows, test_img.cols, CV_8U);
    printf("r %d c %d\n", test_img.rows, test_img.cols);
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
        cv::Mat edges;
        cv::Canny(res, edges, 0, 1);
        cv::normalize(res, tmp2, 0, 255, cv::NORM_MINMAX);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::imshow("F", tmp2);
        cv::pyrUp(edges, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::pyrUp(tmp2, tmp2);
        cv::imshow("E", tmp2);
        printf("i %d\n", i);
        int sti = (i /(fin_edges.cols/8))*8;
        int stj = (i %(fin_edges.cols/8))*8;
        for(int ii = 0; ii < 8;ii++){
            for(int  jj = 0; jj < 8;jj++){
                fin_edges.at<uchar>(ii + sti, jj + stj) = edges.at<uchar>(ii, jj);
            }
        }
        cv::imshow("RR", fin_edges);
        cv::waitKey();
    }

    return 0;
}
