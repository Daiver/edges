#include <stdio.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

int selectFeaturesFromPatches(std::vector<cv::Mat>& images, std::vector<int> *hs){
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
    cv::PCA pca(zs, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.85);
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
    int boundary = std::min(pca.eigenvectors.rows, 10);
    std::vector<int> hs2(images.size(), 0);
    for(int i = 0; i < images.size(); i++){
        //for(int j = 0; j < pca.eigenvectors.rows; j++){
        for(int j = 0; j < boundary; j++){
            hs2.at(i) += (zs2.at<float>(i, j) < 0) * pow(2, j);
        }
    }
    int hash_table_size = pow(2, boundary);
    int *hash_table = new int[hash_table_size];
    memset(hash_table, -1, sizeof(int) * hash_table_size);
    int counter = 0;
    for(auto x : hs2){
        if (hash_table[x] == -1){
            hash_table[x] = counter;
            counter++;
        }
    }
    for(int i = 0 ; i < hs2.size(); i++){
        hs->at(i) = hash_table[hs2[i]];
    }
    printf("counter %d\n", counter);
    delete [] hash_table;
    return counter;
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
    printf("%zu \n", patches.size());
    std::vector<int> hs(patches.size(), 0);
    int num_of_classes = selectFeaturesFromPatches(patches, &hs);
    for(int i = 0; i < num_of_classes; i++){
        //if(hs[i] == 0) continue;
        //cv::Mat tmp1;
        //cv::pyrUp(patches[i]*5, tmp1);
        //cv::pyrUp(tmp1, tmp1);
        //cv::pyrUp(tmp1, tmp1);
        //cv::imshow("o", tmp1);
        for(int j = 0; j < hs.size(); j++){
            if (i != hs[j]) continue;
            char name[100];
            sprintf(name, "a %d", j);
            cv::Mat tmp2;
            cv::normalize(patches[j], tmp2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::pyrUp(tmp2, tmp2);
            cv::pyrUp(tmp2, tmp2);
            cv::normalize(tmp2, tmp2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::imshow(name, tmp2);
            printf(">%d\n", i);
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

