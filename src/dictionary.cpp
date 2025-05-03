#include "dictionary.h"
#include "feature_extractor.h"

DictionaryBuilder::DictionaryBuilder(int dictionarySize, int maxIterations) 
    : dictionarySize(dictionarySize), maxIterations(maxIterations), useFlann(false) {
}

void DictionaryBuilder::buildDictionary(const std::vector<std::string>& featureFiles) {
    // Load all descriptors
    std::cout << "Loading descriptors from " << featureFiles.size() << " files..." << std::endl;
    std::vector<cv::Mat> allDescriptorsList;
    size_t totalDescriptors = 0;
    
    FeatureExtractor extractor;
    
    for (const auto& file : featureFiles) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        if (extractor.loadFeatures(file, keypoints, descriptors)) {
            if (!descriptors.empty()) {
                allDescriptorsList.push_back(descriptors);
                totalDescriptors += descriptors.rows;
            }
        }
    }
    
    std::cout << "Total descriptors loaded: " << totalDescriptors << std::endl;
    
    // Combine all descriptors into a single matrix
    cv::Mat allDescriptors;
    if (totalDescriptors > 0) {
        // Only use a subset if there are too many descriptors
        const size_t maxDescriptorsForClustering = 100000;  // Adjust based on available memory
        
        if (totalDescriptors > maxDescriptorsForClustering) {
            std::cout << "Using a subset of " << maxDescriptorsForClustering << " descriptors for clustering" << std::endl;
            
            // Create a random subset
            allDescriptors = cv::Mat(maxDescriptorsForClustering, allDescriptorsList[0].cols, allDescriptorsList[0].type());
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> fileDistr(0, allDescriptorsList.size() - 1);
            
            for (size_t i = 0; i < maxDescriptorsForClustering; i++) {
                int fileIdx = fileDistr(gen);
                cv::Mat& descriptors = allDescriptorsList[fileIdx];
                std::uniform_int_distribution<> rowDistr(0, descriptors.rows - 1);
                int rowIdx = rowDistr(gen);
                
                // Copy the descriptor to our subset
                descriptors.row(rowIdx).copyTo(allDescriptors.row(i));
            }
        } else {
            // Use all descriptors
            allDescriptors = cv::Mat(totalDescriptors, allDescriptorsList[0].cols, allDescriptorsList[0].type());
            
            int currentRow = 0;
            for (const auto& descriptors : allDescriptorsList) {
                descriptors.copyTo(allDescriptors.rowRange(currentRow, currentRow + descriptors.rows));
                currentRow += descriptors.rows;
            }
        }
        
        // Perform K-means clustering
        std::cout << "Performing K-means clustering to create dictionary..." << std::endl;
        kMeansClustering(allDescriptors);
        
        std::cout << "Dictionary created with " << dictionary.rows << " visual words" << std::endl;
    } else {
        std::cerr << "Error: No descriptors found in the provided feature files" << std::endl;
    }
}

void DictionaryBuilder::kMeansClustering(const cv::Mat& allDescriptors) {
    // Initialize centroids by randomly selecting k descriptors
    int k = std::min(dictionarySize, allDescriptors.rows);
    cv::Mat centroids(k, allDescriptors.cols, allDescriptors.type());
    
    std::vector<int> indices(allDescriptors.rows);
    for (int i = 0; i < allDescriptors.rows; i++) {
        indices[i] = i;
    }
    
    // Random shuffle
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Select the first k indices
    for (int i = 0; i < k; i++) {
        allDescriptors.row(indices[i]).copyTo(centroids.row(i));
    }
    
    // K-means iterations
    std::vector<int> assignments(allDescriptors.rows, -1);
    std::vector<int> counts(k);
    bool converged = false;
    int iteration = 0;
    
    // If using FLANN, initialize index
    if (useFlann) {
        flannIndex = cv::makePtr<cv::flann::Index>(centroids, cv::flann::KDTreeIndexParams(4));    }
    
    while (!converged && iteration < maxIterations) {
        std::cout << "K-means iteration " << (iteration + 1) << "/" << maxIterations << std::endl;
        
        // Assign descriptors to nearest centroids
        bool changed = false;
        
        // Process in batches to avoid memory issues
        const int batchSize = 10000;
        for (int start = 0; start < allDescriptors.rows; start += batchSize) {
            int end = std::min(start + batchSize, allDescriptors.rows);
            
            for (int i = start; i < end; i++) {
                cv::Mat descriptor = allDescriptors.row(i);
                int nearestCentroid = findNearestCentroid(descriptor, centroids);
                
                if (assignments[i] != nearestCentroid) {
                    assignments[i] = nearestCentroid;
                    changed = true;
                }
            }
        }
        
        // Update centroids
        cv::Mat newCentroids = cv::Mat::zeros(k, allDescriptors.cols, allDescriptors.type());
        std::fill(counts.begin(), counts.end(), 0);
        
        for (int i = 0; i < allDescriptors.rows; i++) {
            int centroidIdx = assignments[i];
            cv::add(newCentroids.row(centroidIdx), allDescriptors.row(i), newCentroids.row(centroidIdx));
            counts[centroidIdx]++;
        }
        
        // Normalize
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                newCentroids.row(i) /= counts[i];
            } else {
                // If a centroid has no assigned points, keep the old one
                centroids.row(i).copyTo(newCentroids.row(i));
            }
        }
        
        // Check for empty clusters and reinitialize them
        for (int i = 0; i < k; i++) {
            if (counts[i] == 0) {
                std::cout << "Warning: Empty cluster " << i << ". Reinitializing..." << std::endl;
                
                // Find the cluster with the most points
                int maxIdx = std::max_element(counts.begin(), counts.end()) - counts.begin();
                
                // Find points assigned to that cluster
                std::vector<int> clusterPoints;
                for (int j = 0; j < allDescriptors.rows; j++) {
                    if (assignments[j] == maxIdx) {
                        clusterPoints.push_back(j);
                    }
                }
                
                // Pick a random point from that cluster
                if (!clusterPoints.empty()) {
                    int randomIdx = clusterPoints[gen() % clusterPoints.size()];
                    allDescriptors.row(randomIdx).copyTo(newCentroids.row(i));
                }
            }
        }
        
        // Update centroids
        centroids = newCentroids.clone();
        
        // Update FLANN index if used
        if (useFlann) {
            flannIndex = cv::makePtr<cv::flann::Index>(centroids, cv::flann::KDTreeIndexParams(4));        
        }
        
        // Check for convergence
        converged = !changed;
        iteration++;
    }
    
    dictionary = centroids.clone();
}

int DictionaryBuilder::findNearestCentroid(const cv::Mat& descriptor, const cv::Mat& centroids) {
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
        double minDist = calculateDistance(descriptor, centroids.row(0));
        
        for (int i = 1; i < centroids.rows; i++) {
            double dist = calculateDistance(descriptor, centroids.row(i));
            if (dist < minDist) {
                minDist = dist;
                nearestIdx = i;
            }
        }
        
        return nearestIdx;
    }
}

double DictionaryBuilder::calculateDistance(const cv::Mat& desc1, const cv::Mat& desc2) {
    // Euclidean distance
    double sum = 0;
    for (int i = 0; i < desc1.cols; i++) {
        double diff = desc1.at<float>(0, i) - desc2.at<float>(0, i);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void DictionaryBuilder::saveDictionary(const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    
    if (!outFile) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write dictionary matrix info
    int rows = dictionary.rows;
    int cols = dictionary.cols;
    int type = dictionary.type();
    outFile.write(reinterpret_cast<char*>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<char*>(&cols), sizeof(cols));
    outFile.write(reinterpret_cast<char*>(&type), sizeof(type));
    
    // Write dictionary data
    if (!dictionary.empty()) {
        outFile.write(reinterpret_cast<const char*>(dictionary.data), 
                     dictionary.rows * dictionary.cols * (type == CV_8U ? 1 : 4));
    }
    
    outFile.close();
}

bool DictionaryBuilder::loadDictionary(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    
    if (!inFile) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return false;
    }
    
    // Read dictionary matrix info
    int rows, cols, type;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    inFile.read(reinterpret_cast<char*>(&type), sizeof(type));
    
    // Read dictionary data
    dictionary.create(rows, cols, type);
    if (rows > 0 && cols > 0) {
        inFile.read(reinterpret_cast<char*>(dictionary.data), 
                  rows * cols * (type == CV_8U ? 1 : 4));
    }
    
    inFile.close();
    
    // Initialize FLANN index if using FLANN
    if (useFlann && !dictionary.empty()) {
        flannIndex = cv::makePtr<cv::flann::Index>(dictionary, cv::flann::KDTreeIndexParams(4));    }
    return true;
}