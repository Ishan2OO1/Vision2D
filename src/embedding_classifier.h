#pragma once
#ifndef EMBEDDING_CLASSIFIER_H
#define EMBEDDING_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

// RegionFeatures structure definition
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

// Structure to hold embedding-based training data
struct EmbeddingData {
    std::string objectName;
    std::string imageName;
    cv::Mat embedding;
    cv::Mat originalROI;
    
    EmbeddingData() {}
    EmbeddingData(const std::string& name, const std::string& image, const cv::Mat& emb, const cv::Mat& roi)
        : objectName(name), imageName(image), originalROI(roi) {
        emb.copyTo(embedding);
    }
};

class EmbeddingClassifier {
private:
    std::vector<EmbeddingData> trainingEmbeddings;
    cv::dnn::Net resnetModel;
    bool modelLoaded;
    
    // Distance metrics for embeddings
    double euclideanDistance(const cv::Mat& emb1, const cv::Mat& emb2);
    double cosineDistance(const cv::Mat& emb1, const cv::Mat& emb2);
    double sumSquaredDifference(const cv::Mat& emb1, const cv::Mat& emb2);
    
    // Preprocessing utilities
    cv::Mat preprocessROI(const cv::Mat& roi, int targetSize = 224);
    cv::Mat extractRotatedROI(const cv::Mat& frame, const RegionFeatures& features);
    
public:
    EmbeddingClassifier();
    ~EmbeddingClassifier();
    
    // Model loading and initialization
    bool loadResNetModel(const std::string& modelPath);
    
    // Embedding computation
    cv::Mat computeEmbedding(const cv::Mat& roi);
    
    // Training data management
    void addTrainingEmbedding(const std::string& objectName, const std::string& imageName, 
                             const cv::Mat& frame, const RegionFeatures& features);
    void loadTrainingEmbeddings(const std::string& csvPath);
    void saveTrainingEmbeddings(const std::string& csvPath);
    
    // Classification methods
    std::string classifyOneShot(const cv::Mat& frame, const RegionFeatures& features, double& confidence);
    std::vector<std::pair<std::string, double>> classifyKNearestEmbeddings(
        const cv::Mat& frame, const RegionFeatures& features, int k = 3);
    
    // Performance evaluation
    void evaluatePerformance(const std::vector<RegionFeatures>& testData, 
                           const std::vector<cv::Mat>& testFrames);
    
    // Utility functions
    size_t getTrainingSize() const { return trainingEmbeddings.size(); }
    std::vector<std::string> getClasses() const;
    void visualizeEmbedding(const cv::Mat& embedding, const std::string& windowName = "Embedding");
    
    // Debug and analysis
    void analyzeEmbeddingSpace();
    void saveEmbeddingVisualization(const std::string& outputPath);
};

#endif // EMBEDDING_CLASSIFIER_H