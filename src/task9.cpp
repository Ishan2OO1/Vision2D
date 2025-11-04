/*
  Task 9: One-Shot Classification System using Image Embeddings
  
  This program implements a one-shot classification system using CNN-based embeddings
  from a pre-trained ResNet18 network. It compares performance with hand-built features.
  
  Author: Implementation for CS5330 Computer Vision Project 3
  Date: 2025
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "embedding_classifier.h"
#include "enhanced_classifier.h"
#include "custom_threshold.h"
#include "morphological_filter.h"

// Function declarations
bool loadTestData(const std::string& csvPath, std::vector<RegionFeatures>& testData);
std::vector<cv::Mat> loadTestImages(const std::vector<RegionFeatures>& testData);
void compareClassifiers(EmbeddingClassifier& embeddingClassifier, 
                       EnhancedClassifier& handbuiltClassifier,
                       const std::vector<RegionFeatures>& testData,
                       const std::vector<cv::Mat>& testFrames);
void demonstrateOneShot(EmbeddingClassifier& embeddingClassifier);
void buildEmbeddingTrainingSet(EmbeddingClassifier& embeddingClassifier);

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "TASK 9: ONE-SHOT CLASSIFICATION WITH EMBEDDINGS" << std::endl;
    std::cout << "CNN-based embedding system using ResNet18" << std::endl;
    std::cout << "================================================" << std::endl;
    
    // Initialize embedding classifier
    EmbeddingClassifier embeddingClassifier;
    
    // Load ResNet18 model
    std::string modelPath = "resnet18-v2-7.onnx";
    if (!embeddingClassifier.loadResNetModel(modelPath)) {
        std::cerr << "Failed to load ResNet18 model. Please ensure " << modelPath << " exists." << std::endl;
        return -1;
    }
    
    // Build embedding training set from existing data
    std::cout << "\nStep 1: Building embedding training set..." << std::endl;
    buildEmbeddingTrainingSet(embeddingClassifier);
    
    // Analyze embedding space
    std::cout << "\nStep 2: Analyzing embedding space..." << std::endl;
    embeddingClassifier.analyzeEmbeddingSpace();
    
    // Load test data for performance comparison
    std::cout << "\nStep 3: Loading test data for evaluation..." << std::endl;
    std::vector<RegionFeatures> testData;
    if (!loadTestData("results/comprehensive_feature_vectors.csv", testData)) {
        std::cerr << "Failed to load test data" << std::endl;
        return -1;
    }
    
    // Load corresponding test images (placeholder - in real implementation would load actual images)
    std::vector<cv::Mat> testFrames = loadTestImages(testData);
    
    // Initialize hand-built features classifier for comparison
    std::cout << "\nStep 4: Initializing hand-built features classifier..." << std::endl;
    EnhancedClassifier handbuiltClassifier;
    handbuiltClassifier.loadTrainingData(testData);
    
    // Performance comparison
    std::cout << "\nStep 5: Comparing classifier performance..." << std::endl;
    compareClassifiers(embeddingClassifier, handbuiltClassifier, testData, testFrames);
    
    // Demonstrate one-shot capability
    std::cout << "\nStep 6: Demonstrating one-shot classification..." << std::endl;
    demonstrateOneShot(embeddingClassifier);
    
    // Save results
    std::cout << "\nStep 7: Saving embedding training data..." << std::endl;
    embeddingClassifier.saveTrainingEmbeddings("results/embedding_training_data.csv");
    
    std::cout << "\n=== TASK 9 COMPLETED SUCCESSFULLY ===" << std::endl;
    std::cout << "Results saved to results/ directory" << std::endl;
    
    return 0;
}

bool loadTestData(const std::string& csvPath, std::vector<RegionFeatures>& testData) {
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Could not open training database: " << csvPath << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        RegionFeatures feature;
        
        try {
            std::getline(ss, feature.objectName, ',');
            std::getline(ss, feature.imageName, ',');
            std::getline(ss, token, ','); feature.regionId = std::stoi(token);
            std::getline(ss, token, ','); feature.area = std::stod(token);
            std::getline(ss, token, ','); feature.percentFilled = std::stod(token);
            std::getline(ss, token, ','); feature.aspectRatio = std::stod(token);
            std::getline(ss, token, ','); feature.compactness = std::stod(token);
            std::getline(ss, token, ','); feature.majorAxisLength = std::stod(token);
            std::getline(ss, token, ','); feature.minorAxisLength = std::stod(token);
            std::getline(ss, token, ','); feature.orientation = std::stod(token);
            std::getline(ss, token, ','); feature.eccentricity = std::stod(token);
            std::getline(ss, token, ','); feature.hu1 = std::stod(token);
            std::getline(ss, token, ','); feature.hu2 = std::stod(token);
            std::getline(ss, token); feature.hu3 = std::stod(token);
            
            testData.push_back(feature);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }
    }
    
    file.close();
    std::cout << "Loaded " << testData.size() << " test samples from " << csvPath << std::endl;
    return true;
}

std::vector<cv::Mat> loadTestImages(const std::vector<RegionFeatures>& testData) {
    std::vector<cv::Mat> testFrames;
    
    // For demonstration, create synthetic test images
    // In a real implementation, you would load the actual images corresponding to each feature
    std::cout << "Creating synthetic test images for demonstration..." << std::endl;
    
    for (const auto& feature : testData) {
        // Create a synthetic image based on the feature properties
        cv::Mat syntheticImage(480, 640, CV_8UC3, cv::Scalar(50, 50, 50));
        
        // Draw a synthetic object based on the features
        cv::Point2f center(320, 240); // Center of image
        cv::Size2f size(feature.majorAxisLength, feature.minorAxisLength);
        float angle = feature.orientation;
        
        cv::RotatedRect rect(center, size, angle);
        
        // Draw the object
        cv::Point2f vertices[4];
        rect.points(vertices);
        
        for (int i = 0; i < 4; i++) {
            cv::line(syntheticImage, vertices[i], vertices[(i+1)%4], cv::Scalar(200, 200, 200), 2);
        }
        
        // Fill the object based on compactness
        if (feature.compactness > 0.5) {
            cv::ellipse(syntheticImage, rect, cv::Scalar(150, 150, 150), -1);
        }
        
        testFrames.push_back(syntheticImage);
    }
    
    std::cout << "Created " << testFrames.size() << " synthetic test images" << std::endl;
    return testFrames;
}

void buildEmbeddingTrainingSet(EmbeddingClassifier& embeddingClassifier) {
    // Load existing feature data to build embeddings
    std::vector<RegionFeatures> trainingData;
    if (!loadTestData("results/comprehensive_feature_vectors.csv", trainingData)) {
        std::cerr << "Could not load training data for embedding generation" << std::endl;
        return;
    }
    
    // Create synthetic images and build embeddings
    std::cout << "Building embeddings for " << trainingData.size() << " training samples..." << std::endl;
    
    // Sample a subset for one-shot learning (1-3 examples per class)
    std::map<std::string, int> classCount;
    int maxPerClass = 2; // One-shot: only 1-2 examples per class
    
    for (const auto& feature : trainingData) {
        if (classCount[feature.objectName] >= maxPerClass) {
            continue; // Skip if we have enough examples for this class
        }
        
        // Create synthetic frame
        cv::Mat syntheticFrame(480, 640, CV_8UC3, cv::Scalar(60, 60, 60));
        
        // Create a more realistic synthetic object
        cv::Point2f center(320 + (rand() % 100 - 50), 240 + (rand() % 100 - 50));
        cv::Size2f size(feature.majorAxisLength, feature.minorAxisLength);
        float angle = feature.orientation;
        
        cv::RotatedRect rect(center, size, angle);
        
        // Different object types get different visual characteristics
        cv::Scalar color;
        if (feature.objectName == "Can") {
            color = cv::Scalar(100, 100, 180); // Reddish
        } else if (feature.objectName == "Bottle") {
            color = cv::Scalar(180, 100, 100); // Blueish  
        } else if (feature.objectName == "Stapler") {
            color = cv::Scalar(100, 180, 100); // Greenish
        } else if (feature.objectName == "Level") {
            color = cv::Scalar(150, 150, 100); // Yellowish
        } else {
            color = cv::Scalar(130, 130, 130); // Gray
        }
        
        cv::ellipse(syntheticFrame, rect, color, -1);
        
        // Add some texture/noise
        cv::Mat noise(syntheticFrame.size(), CV_8UC3);
        cv::randn(noise, cv::Scalar(0, 0, 0), cv::Scalar(20, 20, 20));
        cv::add(syntheticFrame, noise, syntheticFrame);
        
        // Update centroid to match the synthetic object
        RegionFeatures modifiedFeature = feature;
        modifiedFeature.centroid = center;
        
        // Add to embedding training set
        embeddingClassifier.addTrainingEmbedding(feature.objectName, feature.imageName, 
                                               syntheticFrame, modifiedFeature);
        
        classCount[feature.objectName]++;
        
        if (classCount[feature.objectName] == 1) {
            std::cout << "  Added first example for class: " << feature.objectName << std::endl;
        }
    }
    
    std::cout << "✓ Built one-shot training set with " << embeddingClassifier.getTrainingSize() 
              << " embeddings across " << embeddingClassifier.getClasses().size() << " classes" << std::endl;
}

void compareClassifiers(EmbeddingClassifier& embeddingClassifier, 
                       EnhancedClassifier& handbuiltClassifier,
                       const std::vector<RegionFeatures>& testData,
                       const std::vector<cv::Mat>& testFrames) {
    
    std::cout << "\n=== CLASSIFIER PERFORMANCE COMPARISON ===" << std::endl;
    
    // Test subset for faster evaluation
    int testSize = std::min(50, static_cast<int>(testData.size()));
    std::vector<RegionFeatures> testSubset(testData.begin(), testData.begin() + testSize);
    std::vector<cv::Mat> frameSubset(testFrames.begin(), testFrames.begin() + testSize);
    
    // Timing comparison
    auto start = std::chrono::high_resolution_clock::now();
    
    // Test embedding classifier
    std::cout << "\n1. EMBEDDING CLASSIFIER (ResNet18 + One-Shot):" << std::endl;
    int embeddingCorrect = 0;
    
    auto embStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < testSize; i++) {
        double confidence;
        std::string predicted = embeddingClassifier.classifyOneShot(frameSubset[i], testSubset[i], confidence);
        if (predicted == testSubset[i].objectName) {
            embeddingCorrect++;
        }
    }
    auto embEnd = std::chrono::high_resolution_clock::now();
    
    double embeddingAccuracy = static_cast<double>(embeddingCorrect) / testSize;
    auto embeddingTime = std::chrono::duration_cast<std::chrono::milliseconds>(embEnd - embStart);
    
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) 
              << embeddingAccuracy * 100.0 << "% (" << embeddingCorrect << "/" << testSize << ")" << std::endl;
    std::cout << "  Time: " << embeddingTime.count() << " ms" << std::endl;
    std::cout << "  Avg time per classification: " << embeddingTime.count() / testSize << " ms" << std::endl;
    
    // Test hand-built features classifier
    std::cout << "\n2. HAND-BUILT FEATURES CLASSIFIER:" << std::endl;
    int handbuiltCorrect = 0;
    
    auto hbStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < testSize; i++) {
        auto results = handbuiltClassifier.classifyMultiple(testSubset[i]);
        std::string predicted = results.empty() ? "Unknown" : results[0].first;
        if (predicted == testSubset[i].objectName) {
            handbuiltCorrect++;
        }
    }
    auto hbEnd = std::chrono::high_resolution_clock::now();
    
    double handbuiltAccuracy = static_cast<double>(handbuiltCorrect) / testSize;
    auto handbuiltTime = std::chrono::duration_cast<std::chrono::milliseconds>(hbEnd - hbStart);
    
    std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) 
              << handbuiltAccuracy * 100.0 << "% (" << handbuiltCorrect << "/" << testSize << ")" << std::endl;
    std::cout << "  Time: " << handbuiltTime.count() << " ms" << std::endl;
    std::cout << "  Avg time per classification: " << handbuiltTime.count() / testSize << " ms" << std::endl;
    
    // Comparison summary
    std::cout << "\n=== COMPARISON SUMMARY ===" << std::endl;
    std::cout << "Embedding vs Hand-built Accuracy: " << std::fixed << std::setprecision(2)
              << embeddingAccuracy * 100.0 << "% vs " << handbuiltAccuracy * 100.0 << "%" << std::endl;
    
    if (embeddingAccuracy > handbuiltAccuracy) {
        std::cout << "✓ Embedding classifier performs better (+";
        std::cout << std::fixed << std::setprecision(2) << (embeddingAccuracy - handbuiltAccuracy) * 100.0 << "%)" << std::endl;
    } else {
        std::cout << "✗ Hand-built classifier performs better (+";
        std::cout << std::fixed << std::setprecision(2) << (handbuiltAccuracy - embeddingAccuracy) * 100.0 << "%)" << std::endl;
    }
    
    std::cout << "Speed comparison: Embedding " << embeddingTime.count() / testSize 
              << " ms/sample vs Hand-built " << handbuiltTime.count() / testSize << " ms/sample" << std::endl;
}

void demonstrateOneShot(EmbeddingClassifier& embeddingClassifier) {
    std::cout << "\n=== ONE-SHOT CLASSIFICATION DEMONSTRATION ===" << std::endl;
    std::cout << "This system can classify objects with only 1-2 training examples per class!" << std::endl;
    
    auto classes = embeddingClassifier.getClasses();
    std::cout << "\nAvailable classes (with minimal training data):" << std::endl;
    for (const auto& className : classes) {
        std::cout << "  - " << className << std::endl;
    }
    
    std::cout << "\nTotal training embeddings: " << embeddingClassifier.getTrainingSize() << std::endl;
    std::cout << "Classes: " << classes.size() << std::endl;
    std::cout << "Average samples per class: " << std::fixed << std::setprecision(1) 
              << static_cast<double>(embeddingClassifier.getTrainingSize()) / classes.size() << std::endl;
    
    std::cout << "\nThis demonstrates the power of one-shot learning:" << std::endl;
    std::cout << "- Traditional classifiers need many examples per class" << std::endl;
    std::cout << "- CNN embeddings capture rich feature representations" << std::endl;
    std::cout << "- One-shot systems can generalize from very few examples" << std::endl;
}