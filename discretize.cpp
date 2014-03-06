#include "discretize.h"

void selectFeaturesFromPatches(std::vector<cv::Mat> images, 
        std::vector<int> *hs, int *num_of_classes, int *seg_idx){
    if(images.size() == 1) {
        hs->at(0) = 0;
        *num_of_classes = 1;
        *seg_idx = 0;
        return;
    }
    printf("start disr %d\n", images.size());
    std::vector<int> indxs;
    for(int i = 0; i < 256; i++){
        int x = (int)rand() % (256*256);
        while((x/256) == (x%256))
            x = (int)rand() % (256*256);
        indxs.push_back(x);
    }
    //std::vector<std::vector<double>> zs(images.size());
    printf("ZS compute\n");
    cv::Mat zs(images.size(), 256, CV_32F);
    for(int i = 0; i < images.size(); i++){
        for(int j = 0; j < 256; j++){
            zs.at<float>(i, j) = 
                images[i].data[indxs[j]/256] == images[i].data[indxs[j]%256];
        }
    }

    printf("deleting indexs\n");
    std::vector<int> indx_to_delete;
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

        sum = 0;
        for(int i = 0; i < images.size(); i++){
            sum += zs.at<float>(i, j);
        }
        if(sum == 0){
            indx_to_delete.push_back(j);
        }
        //printf("\n");
    }
    if(indx_to_delete.size() > 0){
        cv::Mat tmp_zs(images.size(), 256 - indx_to_delete.size(), CV_32F);
        int counter = 0;
        for(int j = 0; j < 256; j++){
            if(std::find(indx_to_delete.begin(), indx_to_delete.end(),j) 
                    == indx_to_delete.end()){
                for(int i = 0; i < images.size(); i++){
                    tmp_zs.at<float>(i, counter) = zs.at<float>(i, j);
                }
                //tmp_zs.col(counter) = zs.col(j);
                counter += 1;
            }
        }
        zs = tmp_zs;
    }
    for(int j = 0; j < 256 - indx_to_delete.size(); j++){
        float sum = 0;
        for(int i = 0; i < images.size(); i++){
            sum += zs.at<float>(i, j);
        }
        if(sum == 0){
            printf("zero col %d\n", j);
        }
    }

    printf("search min\n");
    int min_idx = -1;
    float min_value = 10000;
    for(int i = 0; i < images.size(); i++){
        float sum = 0;
        for(int j = 0; j < zs.cols; j++){
            sum += pow(zs.at<float>(i, j), 2);
        }
        if (sum < min_value){
            min_idx = i;
            min_value = sum;
        }
    }

    if(zs.cols == 0){
        hs->at(0) = 0;
        *num_of_classes = 1;
        *seg_idx = 0;
        return;
    }
    printf("pca... %d %d\n", zs.rows, zs.cols);
    cv::PCA pca(zs, cv::Mat(), CV_PCA_DATA_AS_ROW, 0.82);
    printf("PCA # %d\n", pca.eigenvectors.rows);
    cv::Mat zs2(images.size(), pca.eigenvectors.rows, CV_32F);
    for(int i = 0; i < images.size(); i++){
        auto res = pca.project(zs.row(i));
        for(int j = 0; j < pca.eigenvectors.rows; j++){
            zs2.at<float>(i, j) = res.at<float>(0,j);
        }
        /*double mean = cv::mean(images[i])[0];
        double mean2 = cv::mean(zs.row(i))[0];
        double mean3 = cv::mean(zs2.row(i))[0];
        printf("+ %f %f %f\n", mean/images[i].at<uchar>(0,0), mean2, mean3);
        for(int j = 0; j < 10; j++){
            printf("%f ", zs.at<float>(i, j));
        }
        printf("\n");*/
        //printf("%d>", i);
        //for(int j = 0; j < zs2.row(i).cols; j++){
        //    printf("%f ", zs2.at<float>(i, j));
        //}
        //printf("\n");
    }
    int boundary = std::min(pca.eigenvectors.rows, 5);
    std::vector<int> hs2(images.size(), 0);
    for(int i = 0; i < images.size(); i++){
        //for(int j = 0; j < pca.eigenvectors.rows; j++){
        for(int j = 0; j < boundary; j++){
            hs2.at(i) += (zs2.at<float>(i, j) < 0) * pow(2, j);
        }
    }
    int hash_table_size = pow(2, boundary);
    int *hash_table = new int[hash_table_size];
    for(int i = 0; i < hash_table_size; i++) hash_table[i] = -1;
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
    *num_of_classes = counter;
    *seg_idx = min_idx;
    //return counter;
}


