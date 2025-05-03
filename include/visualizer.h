#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "matcher.h"

class Visualizer {
public:
    Visualizer(const std::string& imageDir);
    
    // Create HTML visualization of query and matches
    void createMatchVisualization(const std::string& queryImage,
                                 const std::vector<ImageMatch>& matches,
                                 const std::string& outputHtml);

private:
    std::string imageDir;
    
    // Get image path for a given image name
    std::string getImagePath(const std::string& imageName);
};