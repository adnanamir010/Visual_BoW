# Bag of Visual Words (BoVW) Image Retrieval System

A C++ implementation of the Bag of Visual Words algorithm for visual place recognition and image retrieval.

## Overview

This project implements a complete Bag of Visual Words (BoVW) pipeline for image matching and retrieval. The system can find visually similar images to a query image from a database of images.

Key features:
- SIFT feature extraction from images
- Visual dictionary creation using K-means clustering
- Histogram generation for compact image representation
- TF-IDF weighting to improve discrimination
- Cosine distance-based image matching
- HTML visualization of results

## Requirements

- C++17 compatible compiler
- OpenCV 4.x
- CMake 3.10+

## Building the Project

```bash
# Clone the repository
git clone https://github.com/adnanamir010/Visual_BoW.git
cd Visual_BoW

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --config Release
```

## Usage

The program can be used in a step-by-step manner or as a complete pipeline:

### Step-by-Step Usage

#### 1. Extract SIFT features from images:
```bash
./bovw extract <image_directory> <output_feature_directory>
```

#### 2. Build visual dictionary:
```bash
./bovw build_dict <feature_directory> <output_dictionary_file> [dictionary_size=1000]
```

#### 3. Generate histograms:
```bash
./bovw compute_hist <feature_directory> <dictionary_file> <output_histogram_directory>
```

#### 4. Apply TF-IDF weighting:
```bash
./bovw compute_tfidf <histogram_directory> <output_tfidf_directory> [idf_file]
```

#### 5. Query and visualize results:
```bash
./bovw query <tfidf_directory> <query_file> <output_html_file> [num_matches=10]
```

### Complete Pipeline

Run the entire pipeline with a single command:
```bash
./bovw pipeline <image_directory> <query_image> <output_html_file>
```

## Example

```bash
# Extract features
./bovw extract ./data/images/ ./data/features/

# Build dictionary with 1000 visual words
./bovw build_dict ./data/features/ ./data/dictionary.dat 1000

# Generate histograms
./bovw compute_hist ./data/features/ ./data/dictionary.dat ./data/histograms/

# Apply TF-IDF weighting
./bovw compute_tfidf ./data/histograms/ ./data/histograms_tfidf/

# Query with a specific image
./bovw query ./data/histograms_tfidf/ ./data/histograms_tfidf/imageCompressedCam0_00000123.tfidf ./results/results.html 10

# Or use the complete pipeline
./bovw pipeline ./data/images/ ./data/images/imageCompressedCam0_00000123.png ./results/results.html
```

## Implementation Details

- `feature_extractor.*` - Extracts SIFT features from images
- `dictionary_builder.*` - Builds visual dictionary using K-means clustering
- `histogram_generator.*` - Generates BoVW histograms from features
- `tf_idf_weighter.*` - Applies TF-IDF weighting to histograms
- `image_matcher.*` - Matches images using cosine distance
- `visualizer.*` - Creates HTML visualizations of matches
- `main.cpp` - Main application and command-line interface


## Acknowledgments

- The Bag of Visual Words implementation is based on concepts from Sivic and Zisserman's paper
- SIFT feature extraction uses OpenCV's implementation