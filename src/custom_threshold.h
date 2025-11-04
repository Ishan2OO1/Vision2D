#ifndef CUSTOM_THRESHOLD_H
#define CUSTOM_THRESHOLD_H

#include <opencv2/opencv.hpp>
#include <vector>

class CustomThreshold {
public:
    // Constructor
    CustomThreshold();

    // Core threshold methods (implemented from scratch)
    cv::Mat fixedThreshold(const cv::Mat& input, double threshold, double maxValue = 255.0);
    cv::Mat kmeansThreshold(const cv::Mat& input, int sampleRate = 16);
    cv::Mat saturationBasedThreshold(const cv::Mat& input, double intensityWeight = 0.7, double saturationWeight = 0.3);
    
    // Preprocessing functions
    cv::Mat preprocessImage(const cv::Mat& input, bool blur = true, double blurKernel = 5.0);
    cv::Mat enhanceSaturation(const cv::Mat& input, double factor = 1.5);

    // Utility functions for analysis and debugging
    void displayHistogram(const cv::Mat& input, const std::string& windowName = "Histogram");
    double calculateOtsuThreshold(const cv::Mat& input);

private:
    // Helper functions
    std::pair<double, double> calculateKMeansThreshold(const cv::Mat& input, int sampleRate = 16);
    std::vector<uchar> samplePixels(const cv::Mat& input, int sampleRate);
    double computeVariance(const std::vector<uchar>& data, double mean);
};

#endif // CUSTOM_THRESHOLD_H