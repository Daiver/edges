#ifndef __DISCRETIZE_H__
#define __DISCRETIZE_H__

#include <stdio.h>
#include <string.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void selectFeaturesFromPatches(std::vector<cv::Mat> images, 
                std::vector<int> *hs, int *num_of_classes, int *seg_idx);

#endif
