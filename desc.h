#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

void patchesToVec(cv::Mat img, std::vector<float> *res);
cv::Mat convTri(cv::Mat, float);
