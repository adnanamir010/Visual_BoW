#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <utility>

#include "histogram_gen.h"

struct ImageMatch {
    std::string imageName;
    float distance;
    
    // Add a default constructor
    ImageMatch() : imageName(""), distance(0.0f) {}
    
    // Keep the existing constructor
    ImageMatch(const std::string& name, float dist) : imageName(name), distance(dist) {}
    
    // For sorting (lower distance = better match)
    bool operator<(const ImageMatch& other) const {
        return distance < other.distance;
    }
};

class ImageMatcher {
public:
    ImageMatcher();
    
    // Load all histograms from a directory
    void loadHistograms(const std::vector<std::string>& histogramFiles);
    
    // Find the top N matches for a query histogram
    std::vector<ImageMatch> findMatches(const cv::Mat& queryHistogram, int numMatches = 10);
    
    // Process a query image
    std::vector<ImageMatch> processQuery(const std::string& queryFeatureFile, int numMatches = 10);

private:
    struct ImageHistogram {
        std::string imageName;
        cv::Mat histogram;
    };
    
    std::vector<ImageHistogram> imageHistograms;
    
    // Calculate cosine distance between histograms
    float calculateCosineDistance(const cv::Mat& hist1, const cv::Mat& hist2);
};