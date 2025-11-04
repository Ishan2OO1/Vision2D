#ifndef ENHANCED_CLASSIFIER_H
#define ENHANCED_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>

// Forward declaration of RegionFeatures
struct RegionFeatures;

class EnhancedClassifier {
private:
    std::vector<RegionFeatures> trainingData;
    std::map<std::string, std::vector<double>> featureStats; // mean and std for each feature per class
    std::vector<std::string> objectClasses;
    
    // Calculate feature statistics for normalization
    void calculateFeatureStats();
    
    // Scaled Euclidean distance
    double scaledEuclideanDistance(const RegionFeatures& a, const RegionFeatures& b);
    
    // Manhattan distance with scaling
    double scaledManhattanDistance(const RegionFeatures& a, const RegionFeatures& b);
    
    // Cosine similarity
    double cosineSimilarity(const RegionFeatures& a, const RegionFeatures& b);
    
    // Convert features to vector for calculations
    std::vector<double> featuresToVector(const RegionFeatures& features);
    
public:
    EnhancedClassifier();
    
    // Load training data from vector
    void loadTrainingData(const std::vector<RegionFeatures>& data);
    
    // Classify using scaled Euclidean distance (Task 6 requirement)
    std::string classifyEuclidean(const RegionFeatures& unknown, double& confidence);
    
    // Classify using Manhattan distance
    std::string classifyManhattan(const RegionFeatures& unknown, double& confidence);
    
    // Classify using cosine similarity
    std::string classifyCosine(const RegionFeatures& unknown, double& confidence);
    
    // Multi-classifier approach with voting
    std::vector<std::pair<std::string, double>> classifyMultiple(const RegionFeatures& unknown);
    
    // K-nearest neighbors classification
    std::string classifyKNN(const RegionFeatures& unknown, int k, double& confidence);
    
    // Get all available classes
    std::vector<std::string> getClasses() const { return objectClasses; }
    
    // Get training data size
    size_t getTrainingSize() const { return trainingData.size(); }
    
    // Check if object is unknown (confidence below threshold)
    bool isUnknownObject(const RegionFeatures& unknown, double threshold = 0.3);
};

#endif // ENHANCED_CLASSIFIER_H