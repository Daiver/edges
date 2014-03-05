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

void cutPatchesFromImage(cv::Mat img, std::vector<cv::Mat>* patches);

#endif