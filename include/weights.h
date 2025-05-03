#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_map>

class TfIdfWeighter {
public:
    TfIdfWeighter();
    
    // Calculate IDF values from histograms
    void calculateIdf(const std::vector<std::string>& histogramFiles);
    
    // Apply TF-IDF weighting to a histogram
    cv::Mat applyWeighting(const cv::Mat& histogram);
    
    // Process all histograms in a directory
    void processHistograms(const std::vector<std::string>& histogramFiles, const std::string& outputDir);
    
    // Save and load IDF values
    void saveIdf(const std::string& filename);
    bool loadIdf(const std::string& filename);

private:
    cv::Mat idf;  // IDF values for each visual word
    int totalDocuments;
};