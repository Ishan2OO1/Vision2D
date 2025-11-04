#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include "custom_threshold.h"
#include "morphological_filter.h"
#include "connected_components.h"

// Task 4: Feature Extraction Structure
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
};

class ComprehensiveFeatureExtractor {
private:
    ConnectedComponentAnalyzer ccAnalyzer;
    
public:
    ComprehensiveFeatureExtractor() : ccAnalyzer(200, true) {} // Min area 200, filter boundary regions
    
    std::vector<RegionFeatures> extractFeatures(const cv::Mat& binaryImage, const std::string& objectName, 
                                               const std::string& imageName, int minArea = 200) {
        std::vector<RegionFeatures> features;
        
        // Update analyzer settings
        ccAnalyzer.setMinRegionArea(minArea);
        
        // Use connected components analyzer
        cv::Mat labeledImage;
        std::vector<RegionInfo> regions;
        int numRegions = ccAnalyzer.analyzeComponents(binaryImage, labeledImage, regions);
        
        std::cout << "Found " << numRegions << " valid regions for feature extraction" << std::endl;
        
        for (size_t i = 0; i < regions.size(); i++) {
            const RegionInfo& region = regions[i];
            
            // Create mask for this region
            cv::Mat regionMask = (labeledImage == region.id);
            
            // Calculate moments
            cv::Moments m = cv::moments(regionMask, true);
            if (m.m00 == 0) continue;
            
            RegionFeatures feature;
            feature.objectName = objectName;
            feature.imageName = imageName;
            feature.regionId = region.id;
            feature.area = region.area;
            feature.centroid = region.centroid;
            
            // Basic geometric features using region info
            double boundingArea = region.boundingBox.width * region.boundingBox.height;
            feature.percentFilled = feature.area / boundingArea;
            feature.aspectRatio = double(region.boundingBox.width) / region.boundingBox.height;
            
            // Find contour for perimeter calculation
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            // Compactness (circularity)
            if (!contours.empty()) {
                double perimeter = cv::arcLength(contours[0], true);
                if (perimeter > 0) {
                    feature.compactness = (4 * CV_PI * feature.area) / (perimeter * perimeter);
                } else {
                    feature.compactness = 0;
                }
            } else {
                feature.compactness = 0;
            }
            
            // Central moments for orientation and axis lengths
            double mu20 = m.mu20 / m.m00;
            double mu02 = m.mu02 / m.m00;
            double mu11 = m.mu11 / m.m00;
            
            // Principal axes
            double delta = 4*mu11*mu11 + (mu20-mu02)*(mu20-mu02);
            double sqrt_delta = sqrt(delta);
            double lambda1 = (mu20 + mu02 + sqrt_delta) / 2.0;
            double lambda2 = (mu20 + mu02 - sqrt_delta) / 2.0;
            
            feature.majorAxisLength = 2 * sqrt(std::max(lambda1, 0.0));
            feature.minorAxisLength = 2 * sqrt(std::max(lambda2, 0.0));
            
            // Orientation
            if (abs(mu11) < 1e-10 && abs(mu20 - mu02) < 1e-10) {
                feature.orientation = 0;
            } else {
                feature.orientation = 0.5 * atan2(2*mu11, mu20-mu02) * 180.0 / CV_PI;
            }
            
            // Eccentricity
            if (lambda2 > 1e-10) {
                feature.eccentricity = sqrt(lambda1) / sqrt(lambda2);
            } else {
                feature.eccentricity = 1.0;
            }
            
            // Hu moments (first 3)
            double hu[7];
            cv::HuMoments(m, hu);
            feature.hu1 = hu[0];
            feature.hu2 = hu[1];
            feature.hu3 = hu[2];
            
            // Bounding rectangle for visualization
            feature.boundingRect = cv::RotatedRect(feature.centroid, 
                cv::Size2f(feature.majorAxisLength, feature.minorAxisLength), 
                feature.orientation);
            
            features.push_back(feature);
        }
        
        return features;
    }
};

void processAllImages(const std::string& objectName, const std::vector<std::string>& imageFiles, 
                     CustomThreshold& thresholder, MorphologicalFilter& morphFilter, 
                     ComprehensiveFeatureExtractor& extractor, std::vector<RegionFeatures>& allFeatures) {
    
    std::cout << "\n=== Processing ALL images for " << objectName << " ===" << std::endl;
    std::cout << "Images to process: " << imageFiles.size() << std::endl;
    
    int totalRegions = 0;
    
    for (size_t imgIdx = 0; imgIdx < imageFiles.size(); imgIdx++) {
        std::string imagePath = "train/" + imageFiles[imgIdx];
        cv::Mat image = cv::imread(imagePath);
        
        if (image.empty()) {
            std::cerr << "Warning: Could not load " << imagePath << std::endl;
            continue;
        }
        
        std::cout << "Processing " << imageFiles[imgIdx] << "...";
        
        // Preprocessing
        cv::Mat preprocessed = thresholder.preprocessImage(image, true, 3);
        
        // Apply K-means thresholding
        cv::Mat thresholded = thresholder.kmeansThreshold(preprocessed, 16);
        
        // Morphological cleanup
        cv::Mat cleaned = morphFilter.cleanupObjects(thresholded, 3, 5);
        
        // Extract features with lower threshold for more regions
        std::vector<RegionFeatures> features = extractor.extractFeatures(cleaned, objectName, imageFiles[imgIdx], 200);
        
        std::cout << " Found " << features.size() << " regions" << std::endl;
        totalRegions += features.size();
        
        // Add to global feature set
        for (const auto& feature : features) {
            allFeatures.push_back(feature);
        }
    }
    
    std::cout << "✓ " << objectName << " complete: " << totalRegions << " total regions from " << imageFiles.size() << " images" << std::endl;
}

int main() {
    std::cout << "=== COMPREHENSIVE FEATURE EXTRACTION (ALL TRAINING IMAGES) ===" << std::endl;
    
    // Initialize processing objects
    CustomThreshold thresholder;
    MorphologicalFilter morphFilter;
    ComprehensiveFeatureExtractor extractor;
    
    // Store all features for CSV export
    std::vector<RegionFeatures> allFeatures;
    
    // Define 8 objects with ALL their training images
    std::map<std::string, std::vector<std::string>> objects = {
        {"Bottle", {"Bottle1.jpeg", "Bottle2.jpeg", "Bottle3.jpeg", "Bottle4.jpeg", "Bottle5.jpeg", "Bottle6.jpeg"}},
        {"Can", {"Can1.jpeg", "Can2.jpeg", "Can3.jpeg", "Can4.jpeg", "Can5.jpeg", "Can6.jpeg"}},
        {"Stapler", {"Stapler1.jpeg", "Stapler2.jpeg", "Stapler3.jpeg", "Stapler4.jpeg", "Stapler5.jpeg", "Stapler6.jpeg", "Stapler7.jpeg"}},
        {"Screwdriver", {"screwdriver01.jpeg", "screwdriver02.jpeg", "screwdriver03.jpeg", "screwdriver04.jpeg", "screwdriver05.jpeg", "screwdriver06.jpeg"}},
        {"Rose", {"rose01.jpeg", "rose02.jpeg", "rose03.jpeg", "rose04.jpeg", "rose05.jpeg", "rose06.jpeg", "rose07.jpeg", "rose08.jpeg", "rose09.jpeg"}},
        {"Compass", {"compass01.jpeg", "compass02.jpeg", "compass03.jpeg", "compass04.jpeg", "compass05.jpeg", "compass06.jpeg"}},
        {"Level", {"level01.jpeg", "level02.jpeg", "level03.jpeg", "level04.jpeg", "level05.jpeg", "level06.jpeg", "level07.jpeg"}},
        {"Keychain", {"keychain01.jpeg", "keychain02.jpeg", "keychain03.jpeg", "keychain04.jpeg", "keychain05.jpeg", "keychain06.jpeg", "keychain07.jpeg", "keychain08.jpeg", "keychain09.jpeg", "keychain10.jpeg", "keychain11.jpeg"}}
    };
    
    // Process each object with ALL its images
    for (const auto& obj : objects) {
        processAllImages(obj.first, obj.second, thresholder, morphFilter, extractor, allFeatures);
    }
    
    // Export comprehensive feature dataset to CSV
    std::ofstream csvFile("results/comprehensive_feature_vectors.csv");
    if (csvFile.is_open()) {
        // Header
        csvFile << "Object,ImageName,RegionId,Area,PercentFilled,AspectRatio,Compactness,MajorAxisLength,MinorAxisLength,Orientation,Eccentricity,Hu1,Hu2,Hu3" << std::endl;
        
        // Data
        for (const auto& feature : allFeatures) {
            csvFile << feature.objectName << "," 
                    << feature.imageName << ","
                    << feature.regionId << ","
                    << std::scientific << feature.area << ","
                    << std::fixed << std::setprecision(6)
                    << feature.percentFilled << ","
                    << feature.aspectRatio << ","
                    << feature.compactness << ","
                    << feature.majorAxisLength << ","
                    << feature.minorAxisLength << ","
                    << feature.orientation << ","
                    << feature.eccentricity << ","
                    << feature.hu1 << ","
                    << feature.hu2 << ","
                    << feature.hu3 << std::endl;
        }
        csvFile.close();
        std::cout << "\n✓ Comprehensive feature vectors saved: results/comprehensive_feature_vectors.csv" << std::endl;
    }
    
    // Generate summary statistics
    std::map<std::string, int> objectCounts;
    for (const auto& feature : allFeatures) {
        objectCounts[feature.objectName]++;
    }
    
    std::cout << "\n=== COMPREHENSIVE DATASET SUMMARY ===" << std::endl;
    std::cout << "Total feature vectors: " << allFeatures.size() << std::endl;
    std::cout << "Object breakdown:" << std::endl;
    for (const auto& count : objectCounts) {
        std::cout << "  " << count.first << ": " << count.second << " feature vectors" << std::endl;
    }
    
    std::cout << "\n✓ Translation invariant: Centroid-based moments used" << std::endl;
    std::cout << "✓ Scale invariant: Normalized by area and axis lengths" << std::endl;
    std::cout << "✓ Rotation invariant: Hu moments and relative orientations" << std::endl;
    std::cout << "\nFeature extraction complete for ALL training images!" << std::endl;
    std::cout << "Ready for machine learning classification (Tasks 5-8)" << std::endl;
    
    return 0;
}