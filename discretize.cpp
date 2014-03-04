#include "discretize.h"

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


