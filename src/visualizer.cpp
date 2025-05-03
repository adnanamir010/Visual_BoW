#include "visualizer.h"

Visualizer::Visualizer(const std::string& imageDir) : imageDir(imageDir) {
}

void Visualizer::createMatchVisualization(const std::string& queryImage,
                                         const std::vector<ImageMatch>& matches,
                                         const std::string& outputHtml) {
    std::ofstream outFile(outputHtml);
    
    if (!outFile) {
        std::cerr << "Error: Could not open file for writing: " << outputHtml << std::endl;
        return;
    }
    
    // Write HTML header
    outFile << "<!DOCTYPE html>\n"
            << "<html>\n"
            << "<head>\n"
            << "    <title>BoVW Image Matching Results</title>\n"
            << "    <style>\n"
            << "        body { font-family: Arial, sans-serif; margin: 20px; }\n"
            << "        h1 { color: #333; }\n"
            << "        .query-container { margin-bottom: 20px; }\n"
            << "        .query-image { max-width: 600px; border: 2px solid #333; }\n"
            << "        .matches-container { display: flex; flex-wrap: wrap; }\n"
            << "        .match-item { margin: 10px; text-align: center; }\n"
            << "        .match-image { width: 300px; border: 1px solid #ccc; }\n"
            << "        .match-info { margin-top: 5px; }\n"
            << "    </style>\n"
            << "</head>\n"
            << "<body>\n"
            << "    <h1>Bag of Visual Words - Image Matching Results</h1>\n";
    
    // Query image
    outFile << "    <div class=\"query-container\">\n"
            << "        <h2>Query Image:</h2>\n"
            << "        <img class=\"query-image\" src=\"" << getImagePath(queryImage) << "\" alt=\"Query Image\">\n"
            << "    </div>\n";
    
    // Matches
    outFile << "    <h2>Top " << matches.size() << " Matches:</h2>\n"
            << "    <div class=\"matches-container\">\n";
    
    for (const auto& match : matches) {
        outFile << "        <div class=\"match-item\">\n"
                << "            <img class=\"match-image\" src=\"" << getImagePath(match.imageName) << "\" alt=\"Match\">\n"
                << "            <div class=\"match-info\">\n"
                << "                <p>Image: " << match.imageName << "</p>\n"
                << "                <p>Distance: " << match.distance << "</p>\n"
                << "            </div>\n"
                << "        </div>\n";
    }
    
    outFile << "    </div>\n";
    
    // HTML footer
    outFile << "</body>\n"
            << "</html>\n";
    
    outFile.close();
    
    std::cout << "Created visualization at " << outputHtml << std::endl;
}

std::string Visualizer::getImagePath(const std::string& imageName) {
    // Convert relative path to absolute path if needed
    std::string imagePath = imageDir + "/" + imageName;
    
    // Handle image extensions
    if (std::filesystem::exists(imagePath + ".png")) {
        return imagePath + ".png";
    } else if (std::filesystem::exists(imagePath + ".jpg")) {
        return imagePath + ".jpg";
    } else if (std::filesystem::exists(imagePath + ".jpeg")) {
        return imagePath + ".jpeg";
    } else {
        // Return the path as is, might be a full path already
        return imagePath;
    }
}