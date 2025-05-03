#include "weights.h"
#include "histogram_gen.h"

TfIdfWeighter::TfIdfWeighter() : totalDocuments(0) {
}

void TfIdfWeighter::calculateIdf(const std::vector<std::string>& histogramFiles) {
    std::cout << "Calculating IDF values from " << histogramFiles.size() << " histograms..." << std::endl;
    
    totalDocuments = histogramFiles.size();
    
    if (totalDocuments == 0) {
        std::cerr << "Error: No histogram files provided" << std::endl;
        return;
    }
    
    // First, load one histogram to get the dimensions
    cv::Mat sampleHistogram;
    cv::Mat emptyMat; // Create an empty matrix
    HistogramGenerator histGen(emptyMat);  // Dummy dictionary for loading only
    
    if (!histGen.loadHistogram(histogramFiles[0], sampleHistogram)) {
        std::cerr << "Error: Could not load sample histogram" << std::endl;
        return;
    }
    
    int dictionarySize = sampleHistogram.cols;
    std::vector<int> wordCounts(dictionarySize, 0);
    
    // Count documents containing each word
    for (const auto& file : histogramFiles) {
        cv::Mat histogram;
        
        if (histGen.loadHistogram(file, histogram)) {
            for (int i = 0; i < dictionarySize; i++) {
                if (histogram.at<float>(0, i) > 0) {
                    wordCounts[i]++;
                }
            }
        }
    }
    
    // Calculate IDF
    idf = cv::Mat::zeros(1, dictionarySize, CV_32F);
    for (int i = 0; i < dictionarySize; i++) {
        if (wordCounts[i] > 0) {
            idf.at<float>(0, i) = std::log(static_cast<float>(totalDocuments) / wordCounts[i]);
        }
    }
    
    std::cout << "IDF values calculated for " << dictionarySize << " visual words" << std::endl;
}

cv::Mat TfIdfWeighter::applyWeighting(const cv::Mat& histogram) {
    if (idf.empty() || histogram.cols != idf.cols) {
        std::cerr << "Error: IDF values not calculated or dimensions mismatch" << std::endl;
        return histogram.clone();
    }
    
    // Apply TF-IDF weighting
    cv::Mat weighted = histogram.clone();
    for (int i = 0; i < histogram.cols; i++) {
        weighted.at<float>(0, i) *= idf.at<float>(0, i);
    }
    
    // Normalize
    cv::normalize(weighted, weighted, 1, 0, cv::NORM_L1);
    
    return weighted;
}

void TfIdfWeighter::processHistograms(const std::vector<std::string>& histogramFiles, const std::string& outputDir) {
    std::cout << "Applying TF-IDF weighting to " << histogramFiles.size() << " histograms..." << std::endl;
    
    if (idf.empty()) {
        std::cerr << "Error: IDF values not calculated" << std::endl;
        return;
    }
    
    cv::Mat emptyMat; // Create an empty matrix
    HistogramGenerator histGen(emptyMat);  // Dummy dictionary for loading only
    
    for (const auto& file : histogramFiles) {
        std::string baseName = std::filesystem::path(file).filename().string();
        std::string outputFile = outputDir + "/" + baseName + ".tfidf";
        
        // Check if weighted histogram already exists
        if (std::filesystem::exists(outputFile)) {
            std::cout << "  Weighted histogram already exists for " << baseName << ", skipping..." << std::endl;
            continue;
        }
        
        std::cout << "  Processing: " << baseName << std::endl;
        
        // Load histogram
        cv::Mat histogram;
        
        if (histGen.loadHistogram(file, histogram)) {
            // Apply TF-IDF weighting
            cv::Mat weighted = applyWeighting(histogram);
            
            // Save weighted histogram
            histGen.saveHistogram(outputFile, weighted);
            
            std::cout << "    Applied TF-IDF weighting" << std::endl;
        } else {
            std::cerr << "    Error: Could not load histogram from " << file << std::endl;
        }
    }
}

void TfIdfWeighter::saveIdf(const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    
    if (!outFile) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write total documents
    outFile.write(reinterpret_cast<char*>(&totalDocuments), sizeof(totalDocuments));
    
    // Write IDF matrix info
    int rows = idf.rows;
    int cols = idf.cols;
    int type = idf.type();
    outFile.write(reinterpret_cast<char*>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<char*>(&cols), sizeof(cols));
    outFile.write(reinterpret_cast<char*>(&type), sizeof(type));
    
    // Write IDF data
    if (!idf.empty()) {
        outFile.write(reinterpret_cast<const char*>(idf.data), 
                     idf.rows * idf.cols * sizeof(float));
    }
    
    outFile.close();
}

bool TfIdfWeighter::loadIdf(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    
    if (!inFile) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return false;
    }
    
    // Read total documents
    inFile.read(reinterpret_cast<char*>(&totalDocuments), sizeof(totalDocuments));
    
    // Read IDF matrix info
    int rows, cols, type;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    inFile.read(reinterpret_cast<char*>(&type), sizeof(type));
    
    // Read IDF data
    idf.create(rows, cols, type);
    if (rows > 0 && cols > 0) {
        inFile.read(reinterpret_cast<char*>(idf.data), 
                   rows * cols * sizeof(float));
    }
    
    inFile.close();
    return true;
}