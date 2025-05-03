#include "feature_extractor.h"
#include <filesystem>
namespace fs = std::filesystem;

FeatureExtractor::FeatureExtractor() {
    sift = cv::SIFT::create();
}

std::vector<cv::KeyPoint> FeatureExtractor::extractKeypoints(const cv::Mat& image) {
    std::vector<cv::KeyPoint> keypoints;
    sift->detect(image, keypoints);
    return keypoints;
}

cv::Mat FeatureExtractor::extractDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat descriptors;
    sift->compute(image, keypoints, descriptors);
    return descriptors;
}

void FeatureExtractor::processDirectory(const std::string& imageDir, const std::string& outputDir) {
    std::cout << "Processing images in directory: " << imageDir << std::endl;
    
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
            std::string imagePath = entry.path().string();
            std::string imageName = entry.path().filename().string();
            std::string outputFile = outputDir + "/" + imageName + ".feat";
            
            std::cout << "Processing: " << imageName << std::endl;
            
            // Check if features already exist
            if (fs::exists(outputFile)) {
                std::cout << "  Features already exist, skipping..." << std::endl;
                continue;
            }
            
            // Load image
            cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "  Error: Could not read image" << std::endl;
                continue;
            }
            
            // Extract features
            std::vector<cv::KeyPoint> keypoints = extractKeypoints(image);
            cv::Mat descriptors = extractDescriptors(image, keypoints);
            
            // Save features
            saveFeatures(outputFile, keypoints, descriptors);
            
            std::cout << "  Extracted " << keypoints.size() << " keypoints" << std::endl;
        }
    }
}

void FeatureExtractor::saveFeatures(const std::string& filename, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors) {
    std::ofstream outFile(filename, std::ios::binary);
    
    if (!outFile) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write number of keypoints
    int count = keypoints.size();
    outFile.write(reinterpret_cast<char*>(&count), sizeof(count));
    
    // Write keypoints
    for (const auto& kp : keypoints) {
        outFile.write(reinterpret_cast<const char*>(&kp.pt.x), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&kp.pt.y), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&kp.size), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&kp.angle), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&kp.response), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&kp.octave), sizeof(int));
        outFile.write(reinterpret_cast<const char*>(&kp.class_id), sizeof(int));
    }
    
    // Write descriptor matrix info
    int rows = descriptors.rows;
    int cols = descriptors.cols;
    int type = descriptors.type();
    outFile.write(reinterpret_cast<char*>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<char*>(&cols), sizeof(cols));
    outFile.write(reinterpret_cast<char*>(&type), sizeof(type));
    
    // Write descriptor data
    if (!descriptors.empty()) {
        outFile.write(reinterpret_cast<const char*>(descriptors.data), 
                     descriptors.rows * descriptors.cols * (type == CV_8U ? 1 : 4));
    }
    
    outFile.close();
}

bool FeatureExtractor::loadFeatures(const std::string& filename, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
    std::ifstream inFile(filename, std::ios::binary);
    
    if (!inFile) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return false;
    }
    
    // Read number of keypoints
    int count;
    inFile.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    // Read keypoints
    keypoints.resize(count);
    for (int i = 0; i < count; i++) {
        inFile.read(reinterpret_cast<char*>(&keypoints[i].pt.x), sizeof(float));
        inFile.read(reinterpret_cast<char*>(&keypoints[i].pt.y), sizeof(float));
        inFile.read(reinterpret_cast<char*>(&keypoints[i].size), sizeof(float));
        inFile.read(reinterpret_cast<char*>(&keypoints[i].angle), sizeof(float));
        inFile.read(reinterpret_cast<char*>(&keypoints[i].response), sizeof(float));
        inFile.read(reinterpret_cast<char*>(&keypoints[i].octave), sizeof(int));
        inFile.read(reinterpret_cast<char*>(&keypoints[i].class_id), sizeof(int));
    }
    
    // Read descriptor matrix info
    int rows, cols, type;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    inFile.read(reinterpret_cast<char*>(&type), sizeof(type));
    
    // Read descriptor data
    descriptors.create(rows, cols, type);
    if (rows > 0 && cols > 0) {
        inFile.read(reinterpret_cast<char*>(descriptors.data), 
                  rows * cols * (type == CV_8U ? 1 : 4));
    }
    
    inFile.close();
    return true;
}