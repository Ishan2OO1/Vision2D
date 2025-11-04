#include "connected_components.h"
#include <iostream>
#include <algorithm>
#include <random>

ConnectedComponentAnalyzer::ConnectedComponentAnalyzer(int minArea, bool filterBoundary) 
    : minRegionArea(minArea), filterBoundaryRegions(filterBoundary) {
    std::cout << "Connected Components Analyzer initialized:" << std::endl;
    std::cout << "  Min region area: " << minRegionArea << " pixels" << std::endl;
    std::cout << "  Filter boundary regions: " << (filterBoundaryRegions ? "Yes" : "No") << std::endl;
}

bool ConnectedComponentAnalyzer::checkBoundaryContact(const cv::Rect& bbox, const cv::Size& imageSize) {
    const int margin = 2; // Pixels from edge to consider as boundary contact
    
    return (bbox.x <= margin) || 
           (bbox.y <= margin) || 
           (bbox.x + bbox.width >= imageSize.width - margin) || 
           (bbox.y + bbox.height >= imageSize.height - margin);
}

std::vector<RegionInfo> ConnectedComponentAnalyzer::filterRegions(const cv::Mat& labels, 
                                                                const cv::Mat& stats, 
                                                                const cv::Mat& centroids, 
                                                                const cv::Size& imageSize) {
    std::vector<RegionInfo> validRegions;
    int numLabels = stats.rows;
    
    // Skip label 0 (background)
    for (int i = 1; i < numLabels; i++) {
        RegionInfo region;
        region.id = i;
        region.area = stats.at<int>(i, cv::CC_STAT_AREA);
        
        // Get bounding box
        int left = stats.at<int>(i, cv::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        region.boundingBox = cv::Rect(left, top, width, height);
        
        // Get centroid
        region.centroid = cv::Point2f(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
        
        // Check boundary contact
        region.touchesBoundary = checkBoundaryContact(region.boundingBox, imageSize);
        
        // Apply filters
        bool passesAreaFilter = (region.area >= minRegionArea);
        bool passesBoundaryFilter = !filterBoundaryRegions || !region.touchesBoundary;
        
        region.isValid = passesAreaFilter && passesBoundaryFilter;
        
        if (region.isValid) {
            validRegions.push_back(region);
        }
    }
    
    return validRegions;
}

int ConnectedComponentAnalyzer::analyzeComponents(const cv::Mat& binaryImage, cv::Mat& labeledImage, 
                                                std::vector<RegionInfo>& regions) {
    // Clear previous results
    regions.clear();
    detectedRegions.clear();
    
    if (binaryImage.empty()) {
        std::cerr << "Error: Input binary image is empty!" << std::endl;
        return 0;
    }
    
    // Ensure binary image is single channel
    cv::Mat processImage;
    if (binaryImage.channels() == 3) {
        cv::cvtColor(binaryImage, processImage, cv::COLOR_BGR2GRAY);
    } else {
        processImage = binaryImage.clone();
    }
    
    // Perform connected components analysis using OpenCV
    cv::Mat stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(processImage, labeledImage, stats, centroids, 8, CV_32S);
    
    std::cout << "\n=== CONNECTED COMPONENTS ANALYSIS ===" << std::endl;
    std::cout << "Total components detected: " << numComponents - 1 << " (excluding background)" << std::endl;
    
    // Filter regions based on criteria
    regions = filterRegions(labeledImage, stats, centroids, binaryImage.size());
    detectedRegions = regions; // Store for later use
    
    std::cout << "Valid regions after filtering: " << regions.size() << std::endl;
    
    // Print detailed statistics
    if (!regions.empty()) {
        printRegionStatistics(regions);
    }
    
    return static_cast<int>(regions.size());
}

cv::Mat ConnectedComponentAnalyzer::visualizeRegions(const cv::Mat& originalImage, 
                                                   const cv::Mat& labeledImage, 
                                                   const std::vector<RegionInfo>& regions) {
    cv::Mat visualization = originalImage.clone();
    
    // Convert to color if grayscale
    if (visualization.channels() == 1) {
        cv::cvtColor(visualization, visualization, cv::COLOR_GRAY2BGR);
    }
    
    // Colors for different regions
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),   // Green
        cv::Scalar(255, 0, 0),   // Blue
        cv::Scalar(0, 0, 255),   // Red
        cv::Scalar(255, 255, 0), // Cyan
        cv::Scalar(255, 0, 255), // Magenta
        cv::Scalar(0, 255, 255), // Yellow
        cv::Scalar(128, 0, 128), // Purple
        cv::Scalar(255, 128, 0)  // Orange
    };
    
    for (size_t i = 0; i < regions.size(); i++) {
        const RegionInfo& region = regions[i];
        cv::Scalar color = colors[i % colors.size()];
        
        // Draw bounding box
        cv::rectangle(visualization, region.boundingBox, color, 2);
        
        // Draw centroid
        cv::circle(visualization, region.centroid, 5, color, -1);
        
        // Label the region
        std::string label = "R" + std::to_string(i + 1) + " (" + std::to_string(region.area) + ")";
        cv::Point labelPos(region.boundingBox.x, region.boundingBox.y - 10);
        cv::putText(visualization, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
    
    // Add summary text
    std::string summary = "Regions: " + std::to_string(regions.size());
    cv::putText(visualization, summary, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    return visualization;
}

cv::Mat ConnectedComponentAnalyzer::createColoredRegionMap(const cv::Mat& labeledImage, 
                                                         const std::vector<RegionInfo>& regions) {
    cv::Mat colorMap = cv::Mat::zeros(labeledImage.size(), CV_8UC3);
    
    // Create random colors for each region
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> colorDist(50, 255);
    
    for (const RegionInfo& region : regions) {
        cv::Scalar regionColor(colorDist(gen), colorDist(gen), colorDist(gen));
        
        // Color all pixels belonging to this region
        cv::Mat mask = (labeledImage == region.id);
        colorMap.setTo(regionColor, mask);
    }
    
    return colorMap;
}

void ConnectedComponentAnalyzer::printRegionStatistics(const std::vector<RegionInfo>& regions) {
    if (regions.empty()) {
        std::cout << "No valid regions found." << std::endl;
        return;
    }
    
    std::cout << "\n--- REGION STATISTICS ---" << std::endl;
    
    // Calculate statistics
    int totalArea = 0;
    int minArea = regions[0].area;
    int maxArea = regions[0].area;
    
    for (size_t i = 0; i < regions.size(); i++) {
        const RegionInfo& region = regions[i];
        totalArea += region.area;
        minArea = std::min(minArea, region.area);
        maxArea = std::max(maxArea, region.area);
        
        std::cout << "Region " << (i + 1) << ": "
                  << "Area=" << region.area 
                  << ", Centroid=(" << region.centroid.x << "," << region.centroid.y << ")"
                  << ", BBox=[" << region.boundingBox.x << "," << region.boundingBox.y 
                  << "," << region.boundingBox.width << "," << region.boundingBox.height << "]"
                  << (region.touchesBoundary ? " [BOUNDARY]" : "") << std::endl;
    }
    
    double avgArea = static_cast<double>(totalArea) / regions.size();
    
    std::cout << "\nSummary:" << std::endl;
    std::cout << "  Total regions: " << regions.size() << std::endl;
    std::cout << "  Area range: " << minArea << " - " << maxArea << " pixels" << std::endl;
    std::cout << "  Average area: " << avgArea << " pixels" << std::endl;
    std::cout << "  Total area: " << totalArea << " pixels" << std::endl;
}

RegionInfo ConnectedComponentAnalyzer::findLargestRegion(const std::vector<RegionInfo>& regions) {
    if (regions.empty()) {
        return RegionInfo();
    }
    
    return *std::max_element(regions.begin(), regions.end(), 
                           [](const RegionInfo& a, const RegionInfo& b) {
                               return a.area < b.area;
                           });
}

RegionInfo ConnectedComponentAnalyzer::findMostCentralRegion(const std::vector<RegionInfo>& regions, 
                                                           const cv::Size& imageSize) {
    if (regions.empty()) {
        return RegionInfo();
    }
    
    cv::Point2f imageCenter(imageSize.width / 2.0f, imageSize.height / 2.0f);
    
    return *std::min_element(regions.begin(), regions.end(), 
                           [&imageCenter](const RegionInfo& a, const RegionInfo& b) {
                               float distA = cv::norm(a.centroid - imageCenter);
                               float distB = cv::norm(b.centroid - imageCenter);
                               return distA < distB;
                           });
}