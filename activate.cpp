
#include "defines.h"
#include "common.h"

#include "decisiontree.h"
#include "randomforest.h"
#include "desc.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "convTri.h"
#include "detect.h"

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
            "/home/daiver/u2.png");
            //"/home/daiver/u.png");
            //"/home/daiver/BSR/BSDS500/data/images/test/100099.jpg");
    
    cv::Mat mag, ori;
    gradientMag<uchar>(test_img2, mag, ori, 4, 0.01);
    cv::imshow("1", test_img2);
    printf("show mag %d %d\n", mag.rows, mag.cols);
    cv::imshow("2", mag);
    printf("show ori %d %d\n", ori.rows, ori.cols);
    cv::normalize(ori, ori, 0, 255, cv::NORM_MINMAX);
    cv::imshow("3", ori);
    cv::waitKey();
}

void gradMagTest2(){
    cv::Mat test_img2 = cv::imread(
            "/home/daiver/u2.png");
            //"/home/daiver/u.png");
            //"/home/daiver/BSR/BSDS500/data/images/test/100099.jpg");
    
    cv::Mat mag, ori;
    gradientMag<uchar>(test_img2, mag, ori, 4, 0.01);
    cv::imshow("1", test_img2);
    printf("show mag %d %d\n", mag.rows, mag.cols);
    cv::imshow("2", mag);
    printf("show ori %d %d\n", ori.rows, ori.cols);
    //cv::normalize(ori, ori, 0, 255, cv::NORM_MINMAX);
    cv::imshow("3", ori);


    cv::Mat f1 = cv::Mat::zeros(test_img2.rows, test_img2.cols, CV_32F);
    cv::Mat f2 = cv::Mat::zeros(test_img2.rows, test_img2.cols, CV_32F);
    cv::Mat f3 = cv::Mat::zeros(test_img2.rows, test_img2.cols, CV_32F);
    cv::Mat f4 = cv::Mat::zeros(test_img2.rows, test_img2.cols, CV_32F);
    for(int i = 0; i < test_img2.rows; i++){
        for(int j = 0; j < test_img2.cols; j++){
            float p = mag.at<float>(i, j);
            float f = ori.at<float>(i, j) ;
            //float f = cv::fastAtan2(gradY.at<short>(i, j), gradX.at<short>(i, j));
            //printf("%d %d %f\n", i, j, f);
            if(p > 1.0){
                if(f < 90.0)
                    f1.at<float>(i, j) = p;
                else if(f >= 90.0 && f < 180.0)
                    f2.at<float>(i, j) = p;
                else if(f >= 180.0 && f < 270.0)
                    f3.at<float>(i, j) = p;
                else if(f >= 270.0 && f < 360.0){
                    f4.at<float>(i, j) = p;
                }
            }
            //res->push_back(f);
        }
    }
    cv::imshow("f1", f1);
    cv::imshow("f2", f2);
    cv::imshow("f3", f3);
    cv::imshow("f4", f4);
    cv::waitKey();
    //cv::imshow("f5", f1 + f2 + f3 + f4);
    //cv::imshow("f6", abs(ori - f1 + f2 + f3 + f4));
    //cv::waitKey();
}

void testDesc(){
    cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
    //cv::Mat test_img = cv::imread("/home/daiver/coding/edges/imgs/img/1.jpg");
    //cv::Mat test_img = cv::imread("/home/daiver/u2.png");
    cv::cvtColor(test_img, test_img, CV_BGR2Luv);
    std::vector <float> tmp;
    patchesToVec(test_img, &tmp);
}

int main(int argc, char** argv){
    //convTriTest(); return 0;
    //gradMagTest(); return 0;
    //gradMagTest2(); return 0;
#ifdef DESC_DEBUG_ACT
    testDesc(); return 0;
#endif
    RandomForest tree(8);
    tree.load("../model/forest");

    //cv::Mat test_img = cv::imread("/home/daiver/coding/edges/imgs/img/1.jpg");
    //cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/train/100075.jpg");
    //cv::Mat test_img = cv::imread("/home/daiver/BSR/BSDS500/data/images/test/29030.jpg");
    cv::Mat test_img = cv::imread(argv[1]);
    printf("BEFORE detect");
    cv::Mat test_res = detect2(tree, test_img);
    printf("after detect");
    cv::imshow("ORIG", test_img);
    cv::imshow("rep", test_res);
    cv::imwrite("res.png", test_res);
    cv::waitKey();
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
