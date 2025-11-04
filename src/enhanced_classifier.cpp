#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include "enhanced_classifier.h"

// Ensure RegionFeatures is properly defined
struct RegionFeatures {
    std::string objectName;
    std::string imageName;
    int regionId;
    double area;
    double percentFilled;
    double aspectRatio;
    double compactness;
    double majorAxisLength;
    double minorAxisLength;
    double orientation;
    double eccentricity;
    double hu1, hu2, hu3;
    cv::Point2f centroid;
    cv::RotatedRect boundingRect;
    
    RegionFeatures() : objectName(""), imageName(""), regionId(0), area(0), percentFilled(0), 
                      aspectRatio(0), compactness(0), majorAxisLength(0), minorAxisLength(0), 
                      orientation(0), eccentricity(0), hu1(0), hu2(0), hu3(0), 
                      centroid(0, 0), boundingRect() {}
};

EnhancedClassifier::EnhancedClassifier() {
    // Constructor implementation
}

void EnhancedClassifier::calculateFeatureStats() {
    // Extract unique class names
    std::set<std::string> uniqueClasses;
    for (const auto& sample : trainingData) {
        uniqueClasses.insert(sample.objectName);
    }
    objectClasses = std::vector<std::string>(uniqueClasses.begin(), uniqueClasses.end());
}

double EnhancedClassifier::scaledEuclideanDistance(const RegionFeatures& a, const RegionFeatures& b) {
    std::vector<double> featA = featuresToVector(a);
    std::vector<double> featB = featuresToVector(b);
    
    double distance = 0.0;
    // Improved scaling factors based on feature importance and typical ranges
    std::vector<double> scalingFactors = {
        1e-6,   // area (very large values, scale down significantly)
        2.0,    // percentFilled (important shape feature)
        1.5,    // aspectRatio (key distinguishing feature)
        3.0,    // compactness (very important for shape classification)
        0.01,   // majorAxisLength (large values)
        0.01,   // minorAxisLength (large values)
        0.1,    // orientation (less important for invariant recognition)
        1.0,    // eccentricity (important shape feature)
        10.0,   // hu1 (Hu moments are small, scale up)
        100.0,  // hu2 (even smaller, scale up more)
        1000.0  // hu3 (smallest, scale up most)
    };
    
    for (size_t i = 0; i < featA.size() && i < featB.size() && i < scalingFactors.size(); i++) {
        double diff = (featA[i] - featB[i]) * scalingFactors[i];
        distance += diff * diff;
    }
    return sqrt(distance);
}

double EnhancedClassifier::scaledManhattanDistance(const RegionFeatures& a, const RegionFeatures& b) {
    std::vector<double> featA = featuresToVector(a);
    std::vector<double> featB = featuresToVector(b);
    
    double distance = 0.0;
    // Improved scaling factors matching Euclidean distance
    std::vector<double> scalingFactors = {
        1e-6,   // area (very large values, scale down significantly)
        2.0,    // percentFilled (important shape feature)
        1.5,    // aspectRatio (key distinguishing feature)
        3.0,    // compactness (very important for shape classification)
        0.01,   // majorAxisLength (large values)
        0.01,   // minorAxisLength (large values)
        0.1,    // orientation (less important for invariant recognition)
        1.0,    // eccentricity (important shape feature)
        10.0,   // hu1 (Hu moments are small, scale up)
        100.0,  // hu2 (even smaller, scale up more)
        1000.0  // hu3 (smallest, scale up most)
    };
    
    for (size_t i = 0; i < featA.size() && i < featB.size() && i < scalingFactors.size(); i++) {
        distance += abs(featA[i] - featB[i]) * scalingFactors[i];
    }
    return distance;
}

double EnhancedClassifier::cosineSimilarity(const RegionFeatures& a, const RegionFeatures& b) {
    std::vector<double> featA = featuresToVector(a);
    std::vector<double> featB = featuresToVector(b);
    
    double dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < featA.size() && i < featB.size(); i++) {
        dotProduct += featA[i] * featB[i];
        normA += featA[i] * featA[i];
        normB += featB[i] * featB[i];
    }
    
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dotProduct / (sqrt(normA) * sqrt(normB));
}

std::vector<double> EnhancedClassifier::featuresToVector(const RegionFeatures& features) {
    return {
        features.area, features.percentFilled, features.aspectRatio, 
        features.compactness, features.majorAxisLength, features.minorAxisLength,
        features.orientation, features.eccentricity, features.hu1, features.hu2, features.hu3
    };
}

void EnhancedClassifier::loadTrainingData(const std::vector<RegionFeatures>& data) {
    trainingData = data;
    calculateFeatureStats();
    std::cout << "Loaded " << trainingData.size() << " training samples across " 
              << objectClasses.size() << " classes" << std::endl;
}

std::string EnhancedClassifier::classifyEuclidean(const RegionFeatures& unknown, double& confidence) {
    if (trainingData.empty()) {
        confidence = 0.0;
        return "Unknown";
    }
    
    double minDistance = std::numeric_limits<double>::max();
    std::string bestMatch = "Unknown";
    
    for (const auto& sample : trainingData) {
        double distance = scaledEuclideanDistance(unknown, sample);
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = sample.objectName;
        }
    }
    
    // Convert distance to confidence (0-1)
    confidence = 1.0 / (1.0 + minDistance);
    return bestMatch;
}

std::string EnhancedClassifier::classifyManhattan(const RegionFeatures& unknown, double& confidence) {
    if (trainingData.empty()) {
        confidence = 0.0;
        return "Unknown";
    }
    
    double minDistance = std::numeric_limits<double>::max();
    std::string bestMatch = "Unknown";
    
    for (const auto& sample : trainingData) {
        double distance = scaledManhattanDistance(unknown, sample);
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = sample.objectName;
        }
    }
    
    confidence = 1.0 / (1.0 + minDistance);
    return bestMatch;
}

std::string EnhancedClassifier::classifyCosine(const RegionFeatures& unknown, double& confidence) {
    if (trainingData.empty()) {
        confidence = 0.0;
        return "Unknown";
    }
    
    double maxSimilarity = -1.0;
    std::string bestMatch = "Unknown";
    
    for (const auto& sample : trainingData) {
        double similarity = cosineSimilarity(unknown, sample);
        if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            bestMatch = sample.objectName;
        }
    }
    
    confidence = (maxSimilarity + 1.0) / 2.0; // Convert [-1,1] to [0,1]
    return bestMatch;
}

std::vector<std::pair<std::string, double>> EnhancedClassifier::classifyMultiple(const RegionFeatures& unknown) {
    std::map<std::string, double> classifierResults;
    
    // Run all classifiers
    double conf1, conf2, conf3, conf4;
    std::string result1 = classifyEuclidean(unknown, conf1);
    std::string result2 = classifyManhattan(unknown, conf2);
    std::string result3 = classifyCosine(unknown, conf3);
    std::string result4 = classifyKNN(unknown, 3, conf4);
    
    // Enhanced weighted voting with quality checks
    // Only include results with reasonable confidence
    if (conf1 > 0.1) classifierResults[result1] += conf1 * 0.35; // Euclidean (primary)
    if (conf2 > 0.1) classifierResults[result2] += conf2 * 0.25; // Manhattan
    if (conf3 > 0.1) classifierResults[result3] += conf3 * 0.15; // Cosine
    if (conf4 > 0.1) classifierResults[result4] += conf4 * 0.25; // KNN
    
    // Sort results by confidence
    std::vector<std::pair<std::string, double>> sortedResults;
    for (const auto& result : classifierResults) {
        sortedResults.push_back({result.first, result.second});
    }
    
    std::sort(sortedResults.begin(), sortedResults.end(), 
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    return sortedResults;
}

std::string EnhancedClassifier::classifyKNN(const RegionFeatures& unknown, int k, double& confidence) {
    if (trainingData.empty()) {
        confidence = 0.0;
        return "Unknown";
    }
    
    std::vector<std::pair<double, std::string>> distances;
    
    for (const auto& sample : trainingData) {
        double distance = scaledEuclideanDistance(unknown, sample);
        distances.push_back({distance, sample.objectName});
    }
    
    std::sort(distances.begin(), distances.end());
    
    // Take top k neighbors
    std::map<std::string, int> votes;
    int actualK = std::min(k, (int)distances.size());
    
    for (int i = 0; i < actualK; i++) {
        votes[distances[i].second]++;
    }
    
    // Find majority vote
    int maxVotes = 0;
    std::string bestMatch = "Unknown";
    for (const auto& vote : votes) {
        if (vote.second > maxVotes) {
            maxVotes = vote.second;
            bestMatch = vote.first;
        }
    }
    
    confidence = double(maxVotes) / actualK;
    return bestMatch;
}

bool EnhancedClassifier::isUnknownObject(const RegionFeatures& unknown, double threshold) {
    auto results = classifyMultiple(unknown);
    return results.empty() || results[0].second < threshold;
}