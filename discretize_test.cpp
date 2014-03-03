#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

void read_imgList(const std::string& filename, std::vector<cv::Mat>& images) {
    std::ifstream file(filename.c_str(), std::ifstream::in);
    std::string line;
    while (std::getline(file, line)) {
        images.push_back(cv::imread(line, 0));
    }
}

int main(){
    return 0;
}
