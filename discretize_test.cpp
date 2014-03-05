#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "discretize.h"
#include "common.h"
#include "decisiontree.h"

int main(){
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> patches;
    read_imgList("images.txt", &images);
    for(auto& img : images){
        cv::imshow("", img);
        cutPatchesFromImage(img, &patches);
    }
    printf("%zu \n", patches.size());
    std::vector<int> hs(patches.size(), 0);
    int num_of_classes, seg_idx;
    selectFeaturesFromPatches(patches, &hs, &num_of_classes, &seg_idx);
    printf("seg index %d\n", seg_idx);
    cv::imshow("seg", patches[seg_idx]);
    cv::waitKey();
    for(int i = 0; i < num_of_classes; i++){
        int counter = 0;
        for(int j = 0; j < hs.size(); j++){
            if (i != hs[j]) continue;
            if (counter > 50) continue;
            char name[100];
            sprintf(name, "a %d", j);
            cv::Mat tmp2;
            cv::normalize(patches[j], tmp2, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            cv::pyrUp(tmp2, tmp2);
            cv::pyrUp(tmp2, tmp2);
            cv::imshow(name, tmp2);
            counter += 1;
        }
        printf(">%d\n", i);
        printf("\n");
        cv::waitKey();

        cv::destroyAllWindows();
    }
    return 0;
}
