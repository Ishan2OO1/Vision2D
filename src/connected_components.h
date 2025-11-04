#ifndef CONNECTED_COMPONENTS_H
#define CONNECTED_COMPONENTS_H

#include <opencv2/opencv.hpp>
#include <vector>

// Structure to hold region information
struct RegionInfo {
    int id;
    int area;
    cv::Point2f centroid;
    cv::Rect boundingBox;
    bool touchesBoundary;
    bool isValid;
    
    RegionInfo() : id(0), area(0), centroid(0, 0), boundingBox(), touchesBoundary(false), isValid(false) {}
};

class ConnectedComponentAnalyzer {
private:
    int minRegionArea;
    bool filterBoundaryRegions;
    std::vector<RegionInfo> detectedRegions;
    
    // Check if region touches image boundary
    bool checkBoundaryContact(const cv::Rect& bbox, const cv::Size& imageSize);
    
    // Filter regions based on size and boundary conditions
    std::vector<RegionInfo> filterRegions(const cv::Mat& labels, const cv::Mat& stats, 
                                         const cv::Mat& centroids, const cv::Size& imageSize);

public:
    ConnectedComponentAnalyzer(int minArea = 200, bool filterBoundary = true);
    
    // Main connected components analysis function
    int analyzeComponents(const cv::Mat& binaryImage, cv::Mat& labeledImage, 
                         std::vector<RegionInfo>& regions);
    
    // Visualization functions
    cv::Mat visualizeRegions(const cv::Mat& originalImage, const cv::Mat& labeledImage, 
                           const std::vector<RegionInfo>& regions);
    cv::Mat createColoredRegionMap(const cv::Mat& labeledImage, 
                                  const std::vector<RegionInfo>& regions);
    
    // Utility functions
    void printRegionStatistics(const std::vector<RegionInfo>& regions);
    RegionInfo findLargestRegion(const std::vector<RegionInfo>& regions);
    RegionInfo findMostCentralRegion(const std::vector<RegionInfo>& regions, const cv::Size& imageSize);
    
    // Getters and setters
    void setMinRegionArea(int area) { minRegionArea = area; }
    void setFilterBoundaryRegions(bool filter) { filterBoundaryRegions = filter; }
    int getMinRegionArea() const { return minRegionArea; }
    bool getFilterBoundaryRegions() const { return filterBoundaryRegions; }
};

#endif // CONNECTED_COMPONENTS_H