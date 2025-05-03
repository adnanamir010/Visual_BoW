#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

class DictionaryBuilder {
public:
    DictionaryBuilder(int dictionarySize = 1000, int maxIterations = 100);
    
    // Build dictionary from a set of feature files
    void buildDictionary(const std::vector<std::string>& featureFiles);
    
    // Save and load dictionary
    void saveDictionary(const std::string& filename);
    bool loadDictionary(const std::string& filename);
    
    // Get the dictionary
    cv::Mat getDictionary() const { return dictionary; }

private:
    // K-means clustering implementation
    void kMeansClustering(const cv::Mat& allDescriptors);
    
    // Find nearest centroid for a descriptor
    int findNearestCentroid(const cv::Mat& descriptor, const cv::Mat& centroids);
    
    // Calculate distance between two descriptors
    double calculateDistance(const cv::Mat& desc1, const cv::Mat& desc2);
    
    int dictionarySize;
    int maxIterations;
    cv::Mat dictionary;  // Centroids form the dictionary
    
    // Optional: Use FLANN for faster nearest neighbor search
    bool useFlann;
    cv::Ptr<cv::flann::Index> flannIndex;
};