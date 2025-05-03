#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class FeatureExtractor {
public:
    FeatureExtractor();
    
    // Extract SIFT features from a single image
    std::vector<cv::KeyPoint> extractKeypoints(const cv::Mat& image);
    cv::Mat extractDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    
    // Process a directory of images
    void processDirectory(const std::string& imageDir, const std::string& outputDir);
    
    // Save features to binary file
    void saveFeatures(const std::string& filename, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors);
    
    // Load features from binary file
    bool loadFeatures(const std::string& filename, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

private:
    cv::Ptr<cv::SIFT> sift;
};