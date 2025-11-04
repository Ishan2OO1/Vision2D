#ifndef MORPHOLOGICAL_FILTER_H
#define MORPHOLOGICAL_FILTER_H

#include <opencv2/opencv.hpp>
#include <vector>

class MorphologicalFilter {
public:
    // Structuring element shapes
    enum StructureShape {
        RECTANGULAR,
        ELLIPTICAL,
        CROSS,
        CUSTOM
    };

    // Morphological operations
    enum Operation {
        EROSION,
        DILATION,
        OPENING,
        CLOSING,
        GRADIENT,
        TOPHAT,
        BLACKHAT
    };

    // Constructor
    MorphologicalFilter();

    // Core morphological operations (implemented from scratch)
    cv::Mat erosion(const cv::Mat& input, const cv::Mat& kernel);
    cv::Mat dilation(const cv::Mat& input, const cv::Mat& kernel);
    cv::Mat opening(const cv::Mat& input, const cv::Mat& kernel);
    cv::Mat closing(const cv::Mat& input, const cv::Mat& kernel);

    // Advanced operations
    cv::Mat morphologicalGradient(const cv::Mat& input, const cv::Mat& kernel);
    cv::Mat topHat(const cv::Mat& input, const cv::Mat& kernel);
    cv::Mat blackHat(const cv::Mat& input, const cv::Mat& kernel);

    // Cleaning strategies for specific problems
    cv::Mat removeNoise(const cv::Mat& input, int kernelSize = 3);
    cv::Mat fillHoles(const cv::Mat& input, int kernelSize = 5);
    cv::Mat smoothBoundaries(const cv::Mat& input, int kernelSize = 3);
    cv::Mat cleanupObjects(const cv::Mat& input, int noiseSize = 3, int holeSize = 5);

    // Structuring element creation
    cv::Mat createKernel(StructureShape shape, int size);
    cv::Mat createKernel(int width, int height, StructureShape shape = RECTANGULAR);

    // Analysis functions
    void analyzeImage(const cv::Mat& input, const std::string& imageName = "Image");
    int countObjects(const cv::Mat& binaryImage);
    double calculateNoiseRatio(const cv::Mat& binaryImage);

private:
    // Helper functions for morphological operations
    bool checkFit(const cv::Mat& image, const cv::Mat& kernel, int x, int y, bool foreground = true);
    bool checkHit(const cv::Mat& image, const cv::Mat& kernel, int x, int y, bool foreground = true);
    
    // Kernel operations
    void applyKernel(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel, Operation op);
    
    // Analysis helpers
    std::vector<std::vector<cv::Point>> findContours(const cv::Mat& binaryImage);
    double calculateCompactness(const std::vector<cv::Point>& contour);
};

#endif // MORPHOLOGICAL_FILTER_H