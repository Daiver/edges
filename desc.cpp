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

/*
template <int T>
void gradientMag(cv::Mat img, cv::Mat &M, cv::Mat &O, int normRad, float normConst){
#ifdef GRAD_MAG_DEBUG
    printf("Compute channels\n");
#endif
    int chnls_size = img.channels();//3;//sizeof(chnls)/sizeof(cv::Mat);
    ///cv::Mat chnls[] = {
    //    cv::Mat::zeros(img.rows, img.cols, img.depth()),
    //    cv::Mat::zeros(img.rows, img.cols, img.depth()),
    //    cv::Mat::zeros(img.rows, img.cols, img.depth()),
    //};/
    printf("CHNSK %d\n", chnls_size);
    cv::Mat *chnls = new cv::Mat[chnls_size];
    for(int i = 0; i < chnls_size; i++) {
        chnls[i] = cv::Mat::zeros(img.rows, img.cols, img.depth());
    }


    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.cols; j++){
            cv::Vec3b p = img.at<cv::Vec3b>(i, j);
            for(int k = 0; k < chnls_size; k++){
                chnls[k].at<T>(i, j) = p[k];//WARN
                //chnls[k].at<uchar>(i, j) = p[k];//WARN
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
    if (normRad == 0) {
        M = M1; return;
    }
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
    }
    //O = (3.14 + O)/2.;
}
*/

void patchesToVec(cv::Mat img_o, std::vector<float> *res){
    cv::Mat img = img_o;
    //cv::pyrDown(img_o, img);
    cv::Mat II[] = {
        cv::Mat(img.rows, img.cols, CV_32F), 
        cv::Mat(img.rows, img.cols, CV_32F), 
        cv::Mat(img.rows, img.cols, CV_32F)
    };
    for(int k = 0; k < 3; k++){
        for(int i = 0; i < img.rows; i++){
            for(int j = 0; j < img.cols; j++){
                cv::Vec3b p = img.at<cv::Vec3b>(i, j);
                float t = p[k];
                II[k].at<float>(i, j) = t;
                //if(II[k].at<float>(i, j) != 0) printf("ttt\n");
            }
        }
    }
    // img 32 II 32

    for(int k = 0; k < 3; k++){
        //II[k] = convTri(II[k], 0);
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
        // II 16
    }
#ifdef DESC_DEBUG
    cv::imshow("img", img);
    cv::Mat tmp;
    cv::normalize(II[0], tmp, 0, 255, cv::NORM_MINMAX);
    cv::imshow("i1", tmp);
    cv::normalize(II[1], tmp, 0, 255, cv::NORM_MINMAX);
    cv::imshow("i2", tmp);
    cv::normalize(II[2], tmp, 0, 255, cv::NORM_MINMAX);
    cv::imshow("i3", tmp);
    cv::waitKey();
#endif

    //printf("0 II %d %d\n", II[0].rows, II[0].cols);
    cv::Mat gray;
    //cv::cvtColor(img, gray, CV_Lab2BGR);
    //cv::cvtColor(gray, gray, CV_BGR2GRAY);
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
        //printf("1 %d img %d %d\n", shrink, img.rows, img.cols);
        // img 16
        cv::Mat mag, ori;
        gradientMag<uchar>(img, mag, ori, 4, 0.01);
        //printf("2 %d mag %d %d\n", shrink, mag.rows, mag.cols);
        // mag 16 ori 16
        //cv::magnitude(Sx, Sy, mag);
        //cv::phase(Sx, Sy, ori, true);
        mag = convTri(mag, 2);
        if (shrink > 0) {
            cv::pyrUp(mag, mag);
            cv::pyrUp(ori, ori);
            cv::pyrUp(img, img);
        }
        //printf("3 %d img %d %d\n", shrink, img.rows, img.cols);
        //printf("4 %d mag %d %d\n", shrink, mag.rows, mag.cols);
        // img 32 mag 32 ori 32
        //cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
        cv::pyrDown(mag, mag);
        //printf("5 %d mag %d %d\n", shrink, mag.rows, mag.cols);
        // mag 16
        for(int i = 0; i < mag.rows; i++){
            for(int j = 0; j < mag.cols; j++){
                float p = mag.at<float>(i, j);
                res->push_back(p);
                //res->push_back(round(p));
            }
        }
        cv::pyrUp(mag, mag);
        //printf("6 %d mag %d %d\n", shrink, mag.rows, mag.cols);
        // mag 32
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
        // F 32
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
        //printf("7 %d f %d %d\n", shrink, F[0].rows, F[0].cols);
        //F 16
#ifdef DESC_DEBUG
        cv::imshow("mag", mag);
        cv::imshow("ori", ori);
        cv::imshow("F[0]", F[0]);
        cv::imshow("F[1]", F[1]);
        cv::imshow("F[2]", F[2]);
        cv::imshow("F[3]", F[3]);
        cv::waitKey();
#endif

        
        cv::Mat for_pairwise[] = {II[0],II[1], II[2],mag,f1,f2,f3,f4};//8
        for(int k = ((shrink > 0) ? 3 : 0); k < 8;  k++){
            cv::Mat reduced = cv::Mat::zeros(5,5,CV_32F);
#ifdef DESC_DEBUG
            cv::imshow("",convTri(for_pairwise[k], 8));
            printf("%d\n", k);
            cv::waitKey();
#endif
            cv::resize(convTri(for_pairwise[k], 8), reduced, reduced.size(), 0, 0, cv::INTER_NEAREST);
#ifdef DESC_DEBUG_SIM
            std::cout << reduced << std::endl;
#endif
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
    }
}


void imageChns(cv::Mat img_o, std::vector<cv::Mat> *chnReg, std::vector<cv::Mat> *chnSim){
    std::vector<cv::Mat> res;
    cv::Mat imShrink, img;
    img_o.convertTo(img, CV_32FC3);
    //img = img_o;
    cv::pyrDown(img, imShrink);
    std::vector<cv::Mat> sep_channels;
    cv::split(img, sep_channels);
    cv::split(imShrink, res);
    //printf("sch %d res %d\n", sep_channels.size(), res.size());
    /*for(int k = 0; k < sep_channels.size(); k++){
        sep_channels[k] = convTri(sep_channels[k]);
    }*/
    for(int shr = 0; shr < 2; shr++){
        if(shr == 1) 
            img = imShrink;
        cv::Mat M, O;
        gradientMag<float>(img, M, O, 4, .01);
        cv::Mat d[4];
        for(int k = 0; k < 4; k++){
            d[k] = cv::Mat::zeros(M.rows, M.cols, CV_32F);
        }
        for(int i = 0; i < M.rows; i++){
            for(int j = 0; j < M.cols; j++){
                float p = M.at<float>(i, j);
                float f = O.at<float>(i, j);
                //float f = cv::fastAtan2(gradY.at<short>(i, j), gradX.at<short>(i, j));
                //printf("%d %d %f\n", i, j, f);
                if(p > 1.0){
                    if(f < 90.0)
                        d[0].at<float>(i, j) = p;
                    else if(f >= 90.0 && f < 180.0)
                        d[1].at<float>(i, j) = p;
                    else if(f >= 180.0 && f < 270.0)
                        d[2].at<float>(i, j) = p;
                    else if(f >= 270.0 && f < 360.0)
                        d[3].at<float>(i, j) = p;
                }
            }
        }
        if(shr == 0){
            cv::pyrDown(M,M);
            cv::pyrDown(O,O);
            for(int k = 0; k < 4; k++) cv::pyrDown(d[k], d[k]);
        }

        res.push_back(M);
        //res.push_back(O);
        for(int k = 0; k < 4; k++) res.push_back(d[k]);
    }
    for(auto &im : res){
        chnReg->push_back(convTri(im, 1));
        chnSim->push_back(convTri(im, 4));
    }
}

void chnsToVecs(std::vector<cv::Mat> &chns, std::vector<cv::Mat> &simChns,
        cv::Mat &image,
        cv::Mat &gtruth, 
        std::vector<std::vector<float>> *descs, 
        std::vector<cv::Mat> *gt_patches,
        int n_samples, int p_samples){

    int gt_w = 16;
    int img_w = 32/2;
    //srand(time(NULL));
    const int stride = 4;
    for (int i = 0; i < 2; i++){
        for (int j = 0; j < ((i == 0)?n_samples:p_samples); ){
            int rI = (rand() % chns[0].rows) ;
            int rJ = (rand() % chns[0].cols) ;
            int cI = (rI*2 + img_w);
            int cJ = (rJ*2 + img_w);
            //printf("i %d j ci %d cj %d %d r %d c %d\n", 
            //        rI, rJ, cI, cJ, chns[0].rows, chns[0].cols);
            if(     rI + img_w < chns[0].rows && 
                    rJ + img_w < chns[0].cols &&
                    (cI - gt_w/2) >= 0 && (cI + gt_w/2) < gtruth.rows &&
                    (cJ - gt_w/2) >= 0 && (cJ + gt_w/2) < gtruth.cols){
                cv::Mat gt_tile = gtruth(
                        cv::Range(cI - gt_w/2, cI + gt_w/2),
                        cv::Range(cJ - gt_w/2, cJ + gt_w/2));//.clone();
                if(gt_tile.rows == gt_w && gt_tile.cols == gt_w){
                    cv::Scalar mean, std;
                    cv::meanStdDev(gt_tile, mean, std);
                    //printf("%d %f\n", i, std[0]);
                    if(std[0] != 0.0 && i == 0) {
                        continue;
                    }
                    if(std[0] == 0.0 && i == 1) {
                        continue;
                    }
                    //img_patches->push_back(tileCopy);                
                    //cv::Canny(gt_tile, gt_tile, 0, 1);
                    j++;
                    gt_patches->push_back(gt_tile);
#ifdef CHNS2VECS_DEBUG
                    char name[100];
                    cv::Mat tmp;
                    cv::normalize(gt_tile, tmp, 0, 255, cv::NORM_MINMAX);
                    cv::pyrUp(tmp, tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::imshow("Orig", tmp);
                    cv::meanStdDev(tmp, mean, std);
                    //printf("%d %f\n", i, std[0]);
                    cv::Mat tileCopy = image(
                        cv::Range(2*rI, std::min(2*(rI + img_w), image.rows)),
                        cv::Range(2*rJ, std::min(2*(rJ + img_w), image.cols)));//.clone();
                    cv::normalize(tileCopy, tmp, 0, 255, cv::NORM_MINMAX);
                    cv::pyrUp(tmp, tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::pyrUp(tmp, tmp);
                    cv::imshow("i", tmp);
#endif
                    descs->push_back(std::vector<float>());
                    tileDesc(chns, simChns, rI, rJ, &(descs->at(descs->size() - 1)));
#ifdef CHNS2VECS_DEBUG
                    cv::waitKey();
#endif
                }
            }
        }
    }
}

void tileDesc(
        std::vector<cv::Mat> &chnReg,
        std::vector<cv::Mat> &chnSim,
        int rI, int rJ,
        std::vector<float> *desc
        ){
    const int img_w = 16;
    for(int ch = 0; ch < chnReg.size(); ch++){
        cv::Mat tileCopy = chnReg[ch](
            cv::Range(rI, std::min(rI + img_w, chnReg[ch].rows)),
            cv::Range(rJ, std::min(rJ + img_w, chnReg[ch].cols)));//.clone();
        for(int ii = 0; ii < tileCopy.rows; ii++){
            for(int jj = 0; jj < tileCopy.cols; jj++){
                desc->push_back(tileCopy.at<float>(ii,jj));
            }
        }
        cv::Mat reduced = chnSim[ch](
            cv::Range(rI, std::min(rI + img_w, chnSim[ch].rows)),
            cv::Range(rJ, std::min(rJ + img_w, chnSim[ch].cols)));
        cv::resize(reduced, reduced, cv::Size(5,5));
        for(int i2 = 0; i2 < 25; i2++){
            int x1 = i2/5;
            int y1 = i2%5;
            float p1 = reduced.at<float>(x1, y1);
            for(int j2 = i2; j2 < 25; j2++){
                int x2 = j2/5;
                int y2 = j2%5;
                if(x1 == x2 && y1 == y2) continue;
                float p2 = reduced.at<float>(x2, y2);
                desc->push_back(p1-p2);
            }
        }
    }
}
