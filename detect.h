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

cv::Mat detect(RandomForest &tree, cv::Mat img_o){

    cv::Mat img;
    std::vector<cv::Mat> img_patches, gt_patches;
    cv::cvtColor(img_o, img, CV_BGR2Luv);
    cv::Mat fin_edges = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    //cv::Mat fin_edges = cv::Mat::zeros(img.rows, img.cols, CV_8U);

    int gt_w = 16;
    int img_w = 32;
    int stride = 4;
    for (int i = 0; i < img.rows; i+=stride){
        for (int j = 0; j < img.cols; j+=stride){
            cv::Mat tileCopy = img(
                    cv::Range(i, std::min(i + img_w, img.rows)),
                    cv::Range(j, std::min(j + img_w, img.cols)));//.clone();
            if (tileCopy.rows == img_w && tileCopy.cols == img_w){
                //continue;
                int cI = i + img_w/2;
                int cJ = j + img_w/2;
                std::vector<float> desc;
                patchesToVec(tileCopy, &desc);
                std::vector<cv::Mat> ress = tree.predict(desc);
                auto res = ress[ress.size() - 1];
                //res.convertTo(res, CV_32F);
                //float minE = *std::min_element(res.begin<float>(), res.end<float>());
                //float maxE = *std::max_element(res.begin<float>(), res.end<float>());
                //printf("MM %f %f\n", minE, maxE);
                //res = (res - minE) / maxE;
                //cv::normalize(res, res, 0.0, 1.0, cv::NORM_MINMAX);
                cv::Mat edges, tmp2, tmpO;
                cv::Canny(res, edges, 0, 1);
                //cv::Mat tmp;
                //gradientMag<uchar>(res, edges, tmp, 0, 0.005);
                //gradientMag<float>(res, edges, tmp, 0, 0.005);
                //edges = edges > 0.01;
                //edges = res;
                cv::normalize(res, tmp2, 0, 255, cv::NORM_MINMAX);
                cv::pyrUp(tmp2, tmp2);
                cv::pyrUp(tmp2, tmp2);
                cv::pyrUp(tmp2, tmp2);
                cv::imshow("F", tmp2);
                cv::pyrUp(edges, tmp2);
                cv::pyrUp(tmp2, tmp2);
                cv::pyrUp(tmp2, tmp2);
                cv::imshow("E", tmp2);
                cv::imshow("RR", fin_edges);
                //fin_edges = convTri(fin_edges, 1);
                //std::cout<<edges << std::endl;
                //cv::waitKey();
                //printf("i %d\n", i);
                //int sti = (i /(fin_edges.cols/8))*8;
                //int stj = (i %(fin_edges.cols/8))*8;
                //for(int ii = cI - gt_w/2 ; ii < cI + gt_w/2; ii++){
                //    for(int jj = cJ - gt_w/2 ; jj < cJ + gt_w/2; jj++){//.clone();
                int si = cI - gt_w/2;
                int sj = cJ - gt_w/2;
                if(edges.cols != 16 || edges.rows != 16){printf("Bad edges size %d %d\n", edges.rows, edges.cols);}
                if(edges.channels() > 1) printf("ERR edges ch %d\n", edges.channels());
                //if(edges.depth() != CV_8U) printf("ERR\n");
                for(int ii = 0; ii < 16;ii++){
                    for(int  jj = 0; jj < 16;jj++){
                        //if (edges.at<float>(ii, jj) < 0){
                        //    printf("sub zero %d %d %f\n", ii, jj, edges.at<float>(ii, jj));
                        // }
                            fin_edges.at<float>(ii + si, jj + sj) = edges.at<uchar>(ii, jj);
                            //fin_edges.at<float>(ii + si, jj + sj) = edges.at<float>(ii, jj);
                            //fin_edges.at<uchar>(ii + si, jj + sj) = edges.at<uchar>(ii, jj);
                            //printf("is 1\n");
                        //}
                        //else {
                            //printf("is 0\n");
                        //}
                    }
                }
                //cv::Mat gt_tile = gtruth(
                //        cv::Range(cI - gt_w/2, cI + gt_w/2),
                //        cv::Range(cJ - gt_w/2, cJ + gt_w/2));//.clone();
                //if(gt_tile.rows == gt_w && gt_tile.cols == gt_w){
                //    img_patches->push_back(tileCopy);
                //    //cv::Canny(gt_tile, gt_tile, 1, 2);
                //    gt_patches->push_back(gt_tile);
                //}
            }
        }
    }

    return  convTri(fin_edges, 1);
    //return fin_edges;
}


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
    /*for(int i = 0; i < 8;i++){
        tree.ansamble[i].head->show();
    }*/

    cv::Mat fin_edges = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    for(int i = 0; i < tmp_data.size(); i++){
        std::vector<float> desc;
        patchesToVec(img_patches[i], &desc);
        std::vector<cv::Mat> ress = tree.predict(desc);
        auto res = ress[ress.size() - 1];
        cv::Mat edges, tmp2, tmpO;
        //gradientMag(res, edges, tmpO, 0, 0.005);
        //cv::Sobel(res, edges, CV_32F, 1, 1);
        //edges = edges > 0.1;
        
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
        cv::imshow("RR", fin_edges);
        //fin_edges = convTri(fin_edges, 1);
        //cv::waitKey();
        //printf("i %d\n", i);
        int sti = (i /(fin_edges.cols/8))*8;
        int stj = (i %(fin_edges.cols/8))*8;
        for(int ii = 0; ii < 16;ii++){
            for(int  jj = 0; jj < 16;jj++){
                if (fin_edges.at<uchar>(ii + sti, jj + stj) == 0){
                    fin_edges.at<uchar>(ii + sti, jj + stj) = edges.at<uchar>(ii, jj);
                    //printf("is 1\n");
                }
                else {
                    //printf("is 0\n");
                }
            }
        }
    }
    cv::imshow("RR", fin_edges);
    //fin_edges = convTri(fin_edges, 1);
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
