#include "matcher.h"

ImageMatcher::ImageMatcher() {
}

void ImageMatcher::loadHistograms(const std::vector<std::string>& histogramFiles) {
    std::cout << "Loading " << histogramFiles.size() << " histograms..." << std::endl;
    
    cv::Mat emptyMat;
    imageHistograms.clear();
    HistogramGenerator histGen(emptyMat);  // Dummy dictionary for loading only
    
    for (const auto& file : histogramFiles) {
        std::string imageName = std::filesystem::path(file).stem().string();
        if (imageName.find(".hist") != std::string::npos) {
            imageName = imageName.substr(0, imageName.length() - 5);  // Remove .hist suffix
        }
        
        cv::Mat histogram;
        if (histGen.loadHistogram(file, histogram)) {
            imageHistograms.push_back({imageName, histogram});
        }
    }
    
    std::cout << "Loaded " << imageHistograms.size() << " histograms" << std::endl;
}

std::vector<ImageMatch> ImageMatcher::findMatches(const cv::Mat& queryHistogram, int numMatches) {
    std::vector<ImageMatch> matches;
    
    for (const auto& imageHist : imageHistograms) {
        float distance = calculateCosineDistance(queryHistogram, imageHist.histogram);
        matches.push_back(ImageMatch(imageHist.imageName, distance));
    }
    
    // Sort by distance (ascending)
    std::sort(matches.begin(), matches.end());
    
    // Keep only the top N matches
    if (static_cast<size_t>(numMatches) < matches.size()) {
        matches.resize(numMatches);
    }
    
    return matches;
}

std::vector<ImageMatch> ImageMatcher::processQuery(const std::string& queryFeatureFile, int numMatches) {
    std::cout << "Processing query: " << queryFeatureFile << std::endl;
    
    cv::Mat emptyMat; // Create an empty matrix
    HistogramGenerator histGen(emptyMat);  // Dummy dictionary for loading only
    
    // Check if it's a histogram file
    bool isHistogram = queryFeatureFile.find(".hist") != std::string::npos || 
                      queryFeatureFile.find(".tfidf") != std::string::npos;
    
    cv::Mat queryHistogram;
    if (isHistogram) {
        // Load histogram directly
        if (!histGen.loadHistogram(queryFeatureFile, queryHistogram)) {
            std::cerr << "Error: Could not load query histogram" << std::endl;
            return {};
        }
    } else {
        // Load features and compute histogram
        std::cerr << "Error: Query must be a histogram file (.hist or .tfidf)" << std::endl;
        return {};
    }
    
    // Find matches
    return findMatches(queryHistogram, numMatches);
}

float ImageMatcher::calculateCosineDistance(const cv::Mat& hist1, const cv::Mat& hist2) {
    // Cosine similarity = dot product / (norm(hist1) * norm(hist2))
    float dotProduct = hist1.dot(hist2);
    float norm1 = cv::norm(hist1);
    float norm2 = cv::norm(hist2);
    
    if (norm1 > 0 && norm2 > 0) {
        float cosineSimilarity = dotProduct / (norm1 * norm2);
        return 1.0f - cosineSimilarity;  // Convert similarity to distance
    } else {
        return 1.0f;  // Maximum distance for zero vectors
    }
}