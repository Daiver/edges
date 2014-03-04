#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

void read_imgList(const std::string& filename, std::vector<cv::Mat>* images) {
    std::ifstream file(filename.c_str(), std::ifstream::in);
    std::string line;
    while (std::getline(file, line)) {
        images->push_back(cv::imread(line, 0));
    }
}

void cutPatchesFromImage(cv::Mat img, std::vector<cv::Mat>* patches){
    int w = 16;
    for (int i = 0; i < img.rows; i+=20){
        for (int j = 0; j < img.cols; j+=20){
            cv::Mat tileCopy = img(cv::Range(i, std::min(i + w, img.rows)),
                 cv::Range(j, std::min(j + w, img.cols))).clone();
            if (tileCopy.rows != w || tileCopy.cols != w) continue;
            patches->push_back(tileCopy);
        }
    }
}

/*std::vector<int> getRandIndxs(int size){
    std::vector<int> indxs(size);
    for(int i = 0; i < size; i++) indxs[i] = i;
    for(int i = 0; i < size; i++) swap(&indxs[i], &indxs[rand() % size]);
    return indxs;
}*/

void selectFeaturesFromPatches(std::vector<cv::Mat>& images, std::vector<int> *hs){
    std::vector<int> indxs;
    for(int i = 0; i < 256; i++){
        indxs.push_back((int)(rand() % (256*256)));
    }
    //std::vector<std::vector<double>> zs(images.size());
    cv::Mat zs(images.size(), 256, CV_32F);
    for(int i = 0; i < images.size(); i++){
        for(int j = 0; j < 256; j++){
            zs.at<float>(i, j) = images[i].data[indxs[j]/256] == images[i].data[indxs[j]%256];
            //zs[i].push_back(images[i].data[indxs[j]/256] == images[i].data[indxs[j]%256]);
            //printf("%d %d %d\n", zs[i][j], 
            //        images[i].data[indxs[j] / 256], 
            //        images[i].data[indxs[j] % 256]);
        }
    }
    for(int j = 0; j < 256; j++){
        float sum = 0;
        for(int i = 0; i < images.size(); i++){
            sum += zs.at<float>(i, j);
            //sum += zs[i][j];
            //printf("%0.2f ", zs.at<float>(i, j));
        }
        //printf("\n");
        float norm_cnst = (float)sum/images.size();
        for(int i = 0; i < images.size(); i++){
            //zs[i][j] -= norm_cnst;
            zs.at<float>(i, j) -= norm_cnst;
            //printf("%0.2f ", zs.at<float>(i, j));
        }
        //printf("\n");
    }
    cv::PCA pca(zs, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.81);
    printf("PCA # %d\n", pca.eigenvectors.rows);
    cv::Mat zs2(images.size(), pca.eigenvectors.rows, CV_32F);
    for(int i = 0; i < images.size(); i++){
        zs2.row(i) = pca.project(zs.row(i));
        //for(int j = 0; j < pca.eigenvectors.rows; j++){
        //    //printf("%f ", zs2.at<float>(i, j));
        //}
        //printf("\n\n");
    }

    //std::vector<long> hs(images.size(), 0);
    int boundary = std::min(pca.eigenvectors.rows, 8);
    for(int i = 0; i < images.size(); i++){
        //for(int j = 0; j < pca.eigenvectors.rows; j++){
        for(int j = 0; j < boundary; j++){
            hs->at(i) += (zs2.at<float>(i, j) < 0) * pow(2, j);
        }
        printf("%d\n", hs->at(i));
    }
    
    //for(auto &i : indxs){
    //    printf("%d %d %d\n", i, i / 256, i % 256);
    //} 
}

int main(){
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> patches;
    read_imgList("images.txt", &images);
    for(auto& img : images){
        cv::imshow("", img);
        cutPatchesFromImage(img, &patches);
        //for(auto &tileCopy : patches){
            //cv::imshow("1", tileCopy);
            //cv::waitKey();
        //}
    }
    printf("%d \n", patches.size());
    std::vector<int> hs(patches.size(), 0);
    selectFeaturesFromPatches(patches, &hs);
    for(int i = 0; i < hs.size(); i++){
        for(int j = 0; j < hs.size(); j++){
            if (i == j || hs[i] != hs[j]) continue;
            cv::imshow("o", patches[i]*10);
            char name[100];
            sprintf(name, "a %d", j % 30);
            cv::imshow(name, patches[j]*10);
            printf(">%d\n", hs[i]);
        }
        printf("\n");
        cv::waitKey();
        cv::destroyAllWindows();
    }
    return 0;
}
        /*for(int i = 0 ; i < img.rows; i++){
            for(int j = 0 ; j < img.rows; j++){
                printf("%d ", img.at<uchar>(i, j));
            }
            printf("\n");
        }*/

