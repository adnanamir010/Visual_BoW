#include "histogram_gen.h"

HistogramGenerator::HistogramGenerator(const cv::Mat& dictionary, bool useFlann) 
    : dictionary(dictionary), useFlann(useFlann) {
    
    if (useFlann && !dictionary.empty()) {
        flannIndex = cv::makePtr<cv::flann::Index>(dictionary, cv::flann::KDTreeIndexParams(4));    
    }
}

cv::Mat HistogramGenerator::generateHistogram(const cv::Mat& descriptors) {
    // Create histogram
    cv::Mat histogram = cv::Mat::zeros(1, dictionary.rows, CV_32F);
    
    if (descriptors.empty()) {
        return histogram;
    }
    
    // Assign each descriptor to the nearest visual word
    for (int i = 0; i < descriptors.rows; i++) {
        cv::Mat descriptor = descriptors.row(i);
        int wordIdx = findNearestWord(descriptor);
        histogram.at<float>(0, wordIdx) += 1;
    }
    
    // Normalize histogram
    cv::normalize(histogram, histogram, 1, 0, cv::NORM_L1);
    
    return histogram;
}

void HistogramGenerator::processFeatureFiles(const std::vector<std::string>& featureFiles, const std::string& outputDir) {
    std::cout << "Generating histograms for " << featureFiles.size() << " files..." << std::endl;
    
    FeatureExtractor extractor;
    
    for (const auto& file : featureFiles) {
        std::string baseName = std::filesystem::path(file).filename().string();
        std::string outputFile = outputDir + "/" + baseName + ".hist";
        
        // Check if histogram already exists
        if (std::filesystem::exists(outputFile)) {
            std::cout << "  Histogram already exists for " << baseName << ", skipping..." << std::endl;
            continue;
        }
        
        std::cout << "  Processing: " << baseName << std::endl;
        
        // Load features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        if (extractor.loadFeatures(file, keypoints, descriptors)) {
            if (!descriptors.empty()) {
                // Generate histogram
                cv::Mat histogram = generateHistogram(descriptors);
                
                // Save histogram
                saveHistogram(outputFile, histogram);
                
                std::cout << "    Generated histogram with " << dictionary.rows << " bins" << std::endl;
            } else {
                std::cerr << "    Warning: No descriptors found in " << file << std::endl;
            }
        } else {
            std::cerr << "    Error: Could not load features from " << file << std::endl;
        }
    }
}

int HistogramGenerator::findNearestWord(const cv::Mat& descriptor) {
    if (useFlann) {
        // Use FLANN for approximate nearest neighbor search
        std::vector<float> query(descriptor.ptr<float>(0), 
                                descriptor.ptr<float>(0) + descriptor.cols);
        std::vector<int> indices(1);
        std::vector<float> dists(1);
        
        flannIndex->knnSearch(query, indices, dists, 1, cv::flann::SearchParams(64));
        return indices[0];
    } else {
        // Brute force method
        int nearestIdx = 0;
        double minDist = calculateDistance(descriptor, dictionary.row(0));
        
        for (int i = 1; i < dictionary.rows; i++) {
            double dist = calculateDistance(descriptor, dictionary.row(i));
            if (dist < minDist) {
                minDist = dist;
                nearestIdx = i;
            }
        }
        
        return nearestIdx;
    }
}

double HistogramGenerator::calculateDistance(const cv::Mat& desc1, const cv::Mat& desc2) {
    // Euclidean distance
    double sum = 0;
    for (int i = 0; i < desc1.cols; i++) {
        double diff = desc1.at<float>(0, i) - desc2.at<float>(0, i);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void HistogramGenerator::saveHistogram(const std::string& filename, const cv::Mat& histogram) {
    std::ofstream outFile(filename, std::ios::binary);
    
    if (!outFile) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write histogram matrix info
    int rows = histogram.rows;
    int cols = histogram.cols;
    int type = histogram.type();
    outFile.write(reinterpret_cast<char*>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<char*>(&cols), sizeof(cols));
    outFile.write(reinterpret_cast<char*>(&type), sizeof(type));
    
    // Write histogram data
    if (!histogram.empty()) {
        outFile.write(reinterpret_cast<const char*>(histogram.data), 
                    histogram.rows * histogram.cols * sizeof(float));
    }
    
    outFile.close();
}

bool HistogramGenerator::loadHistogram(const std::string& filename, cv::Mat& histogram) {
    std::ifstream inFile(filename, std::ios::binary);
    
    if (!inFile) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return false;
    }
    
    // Read histogram matrix info
    int rows, cols, type;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    inFile.read(reinterpret_cast<char*>(&type), sizeof(type));
    
    // Read histogram data
    histogram.create(rows, cols, type);
    if (rows > 0 && cols > 0) {
        inFile.read(reinterpret_cast<char*>(histogram.data), 
                   rows * cols * sizeof(float));
    }
    
    inFile.close();
    return true;
}