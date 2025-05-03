#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <chrono>

#include "feature_extractor.h"
#include "dictionary.h"
#include "histogram_gen.h"
#include "weights.h"
#include "matcher.h"
#include "visualizer.h"

namespace fs = std::filesystem;

// Function to get all files with a specific extension in a directory
std::vector<std::string> getFilesWithExtension(const std::string& directory, const std::string& extension) {
    std::vector<std::string> files;
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == extension) {
            files.push_back(entry.path().string());
        }
    }
    
    return files;
}

void printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "  bovw extract <image_dir> <feature_dir>" << std::endl;
    std::cout << "  bovw build_dict <feature_dir> <dictionary_file> [dictionary_size=1000]" << std::endl;
    std::cout << "  bovw compute_hist <feature_dir> <dictionary_file> <histogram_dir>" << std::endl;
    std::cout << "  bovw compute_tfidf <histogram_dir> <tfidf_dir> [idf_file]" << std::endl;
    std::cout << "  bovw query <tfidf_dir> <query_file> <results_html> [num_matches=10]" << std::endl;
    std::cout << "  bovw pipeline <image_dir> <query_image> <results_html>" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "extract") {
        if (argc < 4) {
            std::cout << "Error: Missing arguments for extract command" << std::endl;
            printUsage();
            return 1;
        }
        
        std::string imageDir = argv[2];
        std::string featureDir = argv[3];
        
        // Create feature directory if it doesn't exist
        fs::create_directories(featureDir);
        
        // Extract features from images
        FeatureExtractor extractor;
        std::vector<std::string> imageFiles;
        
        for (const auto& entry : fs::directory_iterator(imageDir)) {
            if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
                imageFiles.push_back(entry.path().string());
            }
        }
        
        extractor.processDirectory(imageDir, featureDir);
        
    } else if (command == "build_dict") {
        if (argc < 4) {
            std::cout << "Error: Missing arguments for build_dict command" << std::endl;
            printUsage();
            return 1;
        }
        
        std::string featureDir = argv[2];
        std::string dictionaryFile = argv[3];
        int dictionarySize = (argc > 4) ? std::stoi(argv[4]) : 1000;
        
        // Get feature files
        std::vector<std::string> featureFiles = getFilesWithExtension(featureDir, ".feat");
        
        if (featureFiles.empty()) {
            std::cerr << "Error: No feature files found in " << featureDir << std::endl;
            return 1;
        }
        
        // Build dictionary
        DictionaryBuilder dictBuilder(dictionarySize);
        dictBuilder.buildDictionary(featureFiles);
        dictBuilder.saveDictionary(dictionaryFile);
        
    } else if (command == "compute_hist") {
        if (argc < 5) {
            std::cout << "Error: Missing arguments for compute_hist command" << std::endl;
            printUsage();
            return 1;
        }
        
        std::string featureDir = argv[2];
        std::string dictionaryFile = argv[3];
        std::string histogramDir = argv[4];
        
        // Create histogram directory if it doesn't exist
        fs::create_directories(histogramDir);
        
        // Get feature files
        std::vector<std::string> featureFiles = getFilesWithExtension(featureDir, ".feat");
        
        if (featureFiles.empty()) {
            std::cerr << "Error: No feature files found in " << featureDir << std::endl;
            return 1;
        }
        
        // Load dictionary
        DictionaryBuilder dictBuilder;
        if (!dictBuilder.loadDictionary(dictionaryFile)) {
            std::cerr << "Error: Could not load dictionary from " << dictionaryFile << std::endl;
            return 1;
        }
        
        // Compute histograms
        HistogramGenerator histGen(dictBuilder.getDictionary(), true);  // Use FLANN for speed
        histGen.processFeatureFiles(featureFiles, histogramDir);
        
    } else if (command == "compute_tfidf") {
        if (argc < 4) {
            std::cout << "Error: Missing arguments for compute_tfidf command" << std::endl;
            printUsage();
            return 1;
        }
        
        std::string histogramDir = argv[2];
        std::string tfidfDir = argv[3];
        std::string idfFile = (argc > 4) ? argv[4] : tfidfDir + "/idf.dat";
        
        // Create TF-IDF directory if it doesn't exist
        fs::create_directories(tfidfDir);
        
        // Get histogram files
        std::vector<std::string> histogramFiles = getFilesWithExtension(histogramDir, ".hist");
        
        if (histogramFiles.empty()) {
            std::cerr << "Error: No histogram files found in " << histogramDir << std::endl;
            return 1;
        }
        
        // Calculate IDF values
        TfIdfWeighter weighter;
        weighter.calculateIdf(histogramFiles);
        weighter.saveIdf(idfFile);
        
        // Apply TF-IDF weighting
        weighter.processHistograms(histogramFiles, tfidfDir);
        
    } else if (command == "query") {
        if (argc < 5) {
            std::cout << "Error: Missing arguments for query command" << std::endl;
            printUsage();
            return 1;
        }
        
        std::string tfidfDir = argv[2];
        std::string queryFile = argv[3];
        std::string resultsHtml = argv[4];
        int numMatches = (argc > 5) ? std::stoi(argv[5]) : 10;
        
        // Get TF-IDF files
        std::vector<std::string> tfidfFiles = getFilesWithExtension(tfidfDir, ".tfidf");
        
        if (tfidfFiles.empty()) {
            std::cerr << "Error: No TF-IDF files found in " << tfidfDir << std::endl;
            return 1;
        }
        
        // Initialize matcher
        ImageMatcher matcher;
        matcher.loadHistograms(tfidfFiles);
        
        // Process query
        std::vector<ImageMatch> matches = matcher.processQuery(queryFile, numMatches);
        
        if (matches.empty()) {
            std::cerr << "Error: No matches found for query" << std::endl;
            return 1;
        }
        
        // Create visualization
        Visualizer visualizer(fs::path(tfidfFiles[0]).parent_path().parent_path().string() + "/images");
        visualizer.createMatchVisualization(fs::path(queryFile).stem().string(), matches, resultsHtml);
        
    } else if (command == "pipeline") {
        if (argc < 5) {
            std::cout << "Error: Missing arguments for pipeline command" << std::endl;
            printUsage();
            return 1;
        }
        
        std::string imageDir = argv[2];
        std::string queryImage = argv[3];
        std::string resultsHtml = argv[4];
        
        // Create temporary directories
        std::string tempDir = "temp_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::string featureDir = tempDir + "/features";
        std::string dictFile = tempDir + "/dictionary.dat";
        std::string histogramDir = tempDir + "/histograms";
        std::string tfidfDir = tempDir + "/tfidf";
        
        fs::create_directories(featureDir);
        fs::create_directories(histogramDir);
        fs::create_directories(tfidfDir);
        
        // 1. Extract features
        std::cout << "Step 1: Extracting features..." << std::endl;
        FeatureExtractor extractor;
        extractor.processDirectory(imageDir, featureDir);
        
        // Extract query features
        cv::Mat queryImg = cv::imread(queryImage, cv::IMREAD_GRAYSCALE);
        if (queryImg.empty()) {
            std::cerr << "Error: Could not read query image" << std::endl;
            return 1;
        }
        
        std::string queryFeatureFile = featureDir + "/query.feat";
        std::vector<cv::KeyPoint> queryKeypoints = extractor.extractKeypoints(queryImg);
        cv::Mat queryDescriptors = extractor.extractDescriptors(queryImg, queryKeypoints);
        extractor.saveFeatures(queryFeatureFile, queryKeypoints, queryDescriptors);
        
        // 2. Build dictionary
        std::cout << "Step 2: Building dictionary..." << std::endl;
        std::vector<std::string> featureFiles = getFilesWithExtension(featureDir, ".feat");
        DictionaryBuilder dictBuilder(1000);
        dictBuilder.buildDictionary(featureFiles);
        dictBuilder.saveDictionary(dictFile);
        
        // 3. Compute histograms
        std::cout << "Step 3: Computing histograms..." << std::endl;
        HistogramGenerator histGen(dictBuilder.getDictionary(), true);
        histGen.processFeatureFiles(featureFiles, histogramDir);
        
        // 4. Calculate TF-IDF weighting
        std::cout << "Step 4: Calculating TF-IDF weighting..." << std::endl;
        std::vector<std::string> histogramFiles = getFilesWithExtension(histogramDir, ".hist");
        TfIdfWeighter weighter;
        weighter.calculateIdf(histogramFiles);
        weighter.processHistograms(histogramFiles, tfidfDir);
        
        // 5. Query image matching
        std::cout << "Step 5: Finding matches for query image..." << std::endl;
        std::vector<std::string> tfidfFiles = getFilesWithExtension(tfidfDir, ".tfidf");
        ImageMatcher matcher;
        matcher.loadHistograms(tfidfFiles);
        
        std::string queryHistFile = histogramDir + "/query.hist";
        std::string queryTfidfFile = tfidfDir + "/query.tfidf";
        
        cv::Mat queryHistogram = histGen.generateHistogram(queryDescriptors);
        histGen.saveHistogram(queryHistFile, queryHistogram);
        
        cv::Mat queryTfidf = weighter.applyWeighting(queryHistogram);
        histGen.saveHistogram(queryTfidfFile, queryTfidf);
        
        std::vector<ImageMatch> matches = matcher.processQuery(queryTfidfFile, 10);
        
        // 6. Visualize results
        std::cout << "Step 6: Creating visualization..." << std::endl;
        Visualizer visualizer(imageDir);
        visualizer.createMatchVisualization("query", matches, resultsHtml);
        
        // Cleanup temporary files
        std::cout << "Cleaning up temporary files..." << std::endl;
        fs::remove_all(tempDir);
        
        std::cout << "Done! Results saved to " << resultsHtml << std::endl;
        
    } else {
        std::cout << "Unknown command: " << command << std::endl;
        printUsage();
        return 1;
    }
    
    return 0;
}