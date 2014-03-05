#ifndef __COMMON_H__
#define __COMMON_H__

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


void read_imgList(const std::string& filename, std::vector<cv::Mat>* images) ;
void read_imgList2(const std::string& filename, std::vector<cv::Mat>* images, 
        std::vector<cv::Mat>* groundTruth) ;

void cutPatchesFromImage(cv::Mat img, std::vector<cv::Mat>* patches);
void cutPatchesFromImage2(cv::Mat img, cv::Mat gtruth, std::vector<cv::Mat>* img_patches, std::vector<cv::Mat> *gt_patches);

#endif
