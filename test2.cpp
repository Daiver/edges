#include "common.h"

#include "decisiontree.h"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

int main(){
    std::vector<cv::Mat> images, gtruth;
    read_imgList2("images2.txt", &images, &gtruth);
    for(int i = 0; i < images.size(); i++){
        cv::imshow("image", images[i]);
        cv::Mat tmp;
        cv::normalize(gtruth[i], tmp, 0, 255, cv::NORM_MINMAX);
        cv::imshow("edges", tmp);
        cv::waitKey();
    }
    return 0;
}
