#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "feature_extractor.h"
#include "dictionary.h"

class HistogramGenerator {
public:
    HistogramGenerator(const cv::Mat& dictionary, bool useFlann = false);
    
    // Generate histogram for a single set of descriptors
    cv::Mat generateHistogram(const cv::Mat& descriptors);
    
    // Process a directory of feature files
    void processFeatureFiles(const std::vector<std::string>& featureFiles, const std::string& outputDir);
    
    // Save and load histograms
    void saveHistogram(const std::string& filename, const cv::Mat& histogram);
    bool loadHistogram(const std::string& filename, cv::Mat& histogram);
    
    // Find nearest visual word for a descriptor
    int findNearestWord(const cv::Mat& descriptor);

private:
    cv::Mat dictionary;
    bool useFlann;
    cv::Ptr<cv::flann::Index> flannIndex;
    
    // Calculate distance between descriptors
    double calculateDistance(const cv::Mat& desc1, const cv::Mat& desc2);
};