#include "desc.h"
#include "defines.h"

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

cv::Mat convTri(cv::Mat img, float r){
    cv::Mat f1, f2, kernel, res;
    if (r <= 1){
        f1 = cv::Mat::zeros(1, 3, CV_32F);
        float p = 12/r/(r + 2) - 2;
        f1.at<float>(0, 0) = 1;
        f1.at<float>(0, 1) = p;
        f1.at<float>(0, 2) = 1;
        f1 /= (2 + p);
    }
    else {
        f1 = cv::Mat::zeros(1, (int)r*2 + 1, CV_32F);
        int i = 0;
        for(; i < (int)r; i++){
            f1.at<float>(0, i) = i + 1;
        }
        f1.at<float>(0, i) = r + 1;
        i++;
        for(; i < (int)r*2 + 1; i++){
            f1.at<float>(0, i) = 2*r - i + 1;
        }
        f1 /= (r+1)*(r + 1);
        //f=[1:r r +1 r:-1:1]/(r + 1)^2
    }
    cv::transpose(f1, f2);
    kernel = f2*f1;
    /*std::cout<<f1 <<"\n\n";
    std::cout<<f2 <<"\n\n";
    std::cout<<kernel;*/
    cv::filter2D(img, res, -1, kernel);
    return res;
}

void gradientMag(cv::Mat img, cv::Mat &M, cv::Mat &O, int normRad, float normConst){
#ifdef GRAD_MAG_DEBUG
    printf("Compute channels\n");
#endif
    cv::Mat chnls[] = {
        cv::Mat::zeros(img.rows, img.cols, img.depth()),
        cv::Mat::zeros(img.rows, img.cols, img.depth()),
        cv::Mat::zeros(img.rows, img.cols, img.depth()),
    };
    int chnls_size = sizeof(chnls)/sizeof(cv::Mat);

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            cv::Vec3b p = img.at<cv::Vec3b>(i, j);
            for(int k = 0; k < chnls_size; k++){
                chnls[k].at<uchar>(i, j) = p[k];//WARN
            }
        }
    }
    cv::Mat Sx[4];
    cv::Mat Sy[4];
    cv::Mat mag[4];
    Sx[3] = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    Sy[3] = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    mag[3] = cv::Mat::zeros(img.rows, img.cols, CV_32F);
#ifdef GRAD_MAG_DEBUG
    printf("Compute mag\n");
#endif
    for(int k = 0; k < chnls_size; k++){
        cv::Sobel(chnls[k], Sx[k], CV_32F, 1, 0);
        cv::Sobel(chnls[k], Sy[k], CV_32F, 0, 1);
        cv::magnitude(Sx[k], Sy[k], mag[k]);
    }
#ifdef GRAD_MAG_DEBUG
    printf("Compute max mag\n");
#endif
    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            float max = 0;
            int ind = 0;
            for(int k = 0; k < chnls_size; k++){
                float p = mag[k].at<float>(i,j);
                if (p > max){
                    max = p;
                    ind = k;
                }
            }
            mag[3].at<float>(i, j) = max;
            Sx[3].at<float>(i, j) = Sx[ind].at<float>(i,j);
            Sy[3].at<float>(i, j) = Sy[ind].at<float>(i,j);
        }
    }
#ifdef GRAD_MAG_DEBUG
    printf("Compute norm\n");
#endif
    cv::Mat M1 = mag[3];
    cv::Mat S = convTri(M1, normRad) + normConst;
    cv::divide(M1, S, M);
    cv::divide(Sx[3], S, Sx[3]);
    cv::divide(Sy[3], S, Sy[3]);
#ifdef GRAD_MAG_DEBUG
    printf("Compute ori\n");
#endif
    O = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::phase(Sx[3], Sy[3], O, true);
    /*for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            if (O.at<float>(i, j) < 0) printf("--- %f\n", O.at<float>(i,j));
            if (O.at<float>(i, j) >3.15) printf("++ %f\n", O.at<float>(i,j));
        }
    }*/
    //O = (3.14 + O)/2.;
}

void patchesToVec(cv::Mat img_o, std::vector<float> *res){
    cv::Mat img = img_o;
    //`cv::pyrDown(img_o, img);
    cv::Mat II[] = {cv::Mat(img.rows, img.cols, CV_32F), 
        cv::Mat(img.rows, img.cols, CV_32F), 
        cv::Mat(img.rows, img.cols, CV_32F)
    };
    for(int k = 0; k < 3; k++){
        for(int i = 0; i < img.rows; i++){
            for(int j = 0; j < img.cols; j++){
                cv::Vec3b p = img.at<cv::Vec3b>(i, j);
                float t = p[k];
                II[k].at<float>(i, j) = t;
            }
        }
    }

    for(int k = 0; k < 3; k++){
        II[k] = convTri(II[k], 0);
        cv::pyrDown(II[k], II[k]);
        for(int i = 0; i < II[k].rows; i++){
            for(int j = 0; j < II[k].cols; j++){
                /*cv::Vec3b p = img.at<cv::Vec3b>(i, j);
                res->push_back(p[0]);
                res->push_back(p[1]);*/
                float p = II[k].at<float>(i, j);
                res->push_back(p);
            }
        }
    }
    cv::Mat gray;
    cv::cvtColor(img, gray, CV_Lab2BGR);
    cv::cvtColor(gray, gray, CV_BGR2GRAY);
    /*cv::Mat gradY;
    cv::Sobel(gray, gradY, CV_16S, 0, 1);
    //cv::normalize(tmp, tmp, 0, 255, cv::NORM_MINMAX);
    cv::Mat gradX;
    cv::Sobel(gray, gradX, CV_16S, 1, 0);
    cv::Mat gradF;
    cv::Sobel(gray, gradF, CV_16S, 1, 1);
    cv::normalize(gradF, gradF, 0, 255, cv::NORM_MINMAX);*/
    /*cv::Mat Sx;
    cv::Sobel(img, Sx, CV_32F, 1, 0, 3);
    cv::Mat Sy;
    cv::Sobel(img, Sy, CV_32F, 0, 1, 3);*/

    for (int shrink = 0; shrink < 2; shrink++){
        if (shrink > 0) cv::pyrDown(img, img);
        cv::Mat mag, ori;
        gradientMag(img, mag, ori, 4, 0.01);
        //cv::magnitude(Sx, Sy, mag);
        //cv::phase(Sx, Sy, ori, true);
        mag = convTri(mag, 2);
        if (shrink > 0) {
            cv::pyrUp(mag, mag);
            cv::pyrUp(ori, ori);
            cv::pyrUp(img, img);
        }
        //cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
        cv::pyrDown(mag, mag);
        for(int i = 0; i < img.rows; i++){
            for(int j = 0; j < img.cols; j++){
                float p = mag.at<float>(i, j);
                res->push_back(p);
                //res->push_back(round(p));
            }
        }
        cv::pyrUp(mag, mag);
        cv::Mat f1 = cv::Mat::zeros(img.rows, img.cols, CV_32F);
        cv::Mat f2 = cv::Mat::zeros(img.rows, img.cols, CV_32F);
        cv::Mat f3 = cv::Mat::zeros(img.rows, img.cols, CV_32F);
        cv::Mat f4 = cv::Mat::zeros(img.rows, img.cols, CV_32F);
        for(int i = 0; i < img.rows; i++){
            for(int j = 0; j < img.cols; j++){
                float p = mag.at<float>(i, j);
                float f = ori.at<float>(i, j);
                //float f = cv::fastAtan2(gradY.at<short>(i, j), gradX.at<short>(i, j));
                //printf("%d %d %f\n", i, j, f);
                if(p > 1.0){
                    if(f < 90.0)
                        f1.at<float>(i, j) = p;
                    else if(f >= 90.0 && f < 180.0)
                        f2.at<float>(i, j) = p;
                    else if(f >= 180.0 && f < 270.0)
                        f3.at<float>(i, j) = p;
                    else if(f >= 270.0 && f < 360.0)
                        f4.at<float>(i, j) = p;
                }
                //res->push_back(f);
            }
        }
        cv::Mat F[] = {f1,f2,f3,f4};
        for(int k = 0; k < 4;  k++){
            F[k] = convTri(F[k], 2);
            cv::pyrDown(F[k], F[k]);
            //cv::normalize(F[k], F[k], 0, 255, cv::NORM_MINMAX);
            for(int i = 0; i < F[k].rows; i++){
                for(int j = 0; j < F[k].cols; j++){
                    float p = F[k].at<float>(i, j);
                    res->push_back(p);
                    //res->push_back(round(p));
                }
            }
        }

        /*
        cv::Mat for_pairwise[] = {II[0],II[1], II[2],mag,f1,f2,f3,f4};//8
        for(int k = ((shrink > 0) ? 3 : 0); k < 8;  k++){
            cv::Mat reduced = cv::Mat::zeros(5,5,CV_32F);
            cv::resize(convTri(for_pairwise[k], 8), reduced, reduced.size());
            for(int i = 0; i < 25; i++){
                int x1 = i/5;
                int y1 = i%5;
                float p1 = for_pairwise[k].at<float>(x1, y1);
                for(int j = i; j < 25; j++){
                    int x2 = j/5;
                    int y2 = j%5;
                    if(x1 == x2 && y1 == y2) continue;
                    float p2 = for_pairwise[k].at<float>(x2, y2);
                    res->push_back(p1 - p2);
                    //res->push_back(round(p1 - p2));
                }
            }
        }
        */
    }
}

