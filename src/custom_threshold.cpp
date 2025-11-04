#include "custom_threshold.h"
#include <algorithm>
#include <random>
#include <iostream>

CustomThreshold::CustomThreshold() {
    // Constructor implementation
}



cv::Mat CustomThreshold::fixedThreshold(const cv::Mat& input, double threshold, double maxValue) {
    cv::Mat gray, result;
    
    // Convert to grayscale if needed
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    // Create output image
    result = cv::Mat::zeros(gray.size(), CV_8UC1);
    
    // Apply threshold manually (from scratch implementation)
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            uchar pixel = gray.at<uchar>(i, j);
            if (pixel < threshold) {
                result.at<uchar>(i, j) = 0;  // Object (black)
            } else {
                result.at<uchar>(i, j) = static_cast<uchar>(maxValue);  // Background (white)
            }
        }
    }
    
    return result;
}

cv::Mat CustomThreshold::kmeansThreshold(const cv::Mat& input, int sampleRate) {
    cv::Mat gray;
    
    // Convert to grayscale if needed
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    // Get K-means threshold using ISODATA algorithm
    auto thresholds = calculateKMeansThreshold(gray, sampleRate);
    double threshold = (thresholds.first + thresholds.second) / 2.0;
    
    std::cout << "K-means threshold calculated: " << threshold << 
                 " (means: " << thresholds.first << ", " << thresholds.second << ")" << std::endl;
    
    // Apply the calculated threshold
    return fixedThreshold(gray, threshold, 255.0);
}

std::pair<double, double> CustomThreshold::calculateKMeansThreshold(const cv::Mat& input, int sampleRate) {
    // Sample pixels (1/sampleRate of total pixels)
    std::vector<uchar> samples = samplePixels(input, sampleRate);
    
    if (samples.empty()) {
        return std::make_pair(64.0, 192.0);  // Default values
    }
    
    // Simple K-means with K=2 (ISODATA algorithm)
    double mean1 = 64.0;   // Initial guess for dark objects
    double mean2 = 192.0;  // Initial guess for light background
    
    const int maxIterations = 50;
    const double tolerance = 1.0;
    
    for (int iter = 0; iter < maxIterations; iter++) {
        std::vector<uchar> cluster1, cluster2;
        
        // Assign pixels to clusters
        for (uchar pixel : samples) {
            double dist1 = std::abs(pixel - mean1);
            double dist2 = std::abs(pixel - mean2);
            
            if (dist1 < dist2) {
                cluster1.push_back(pixel);
            } else {
                cluster2.push_back(pixel);
            }
        }
        
        // Calculate new means
        double newMean1 = mean1, newMean2 = mean2;
        
        if (!cluster1.empty()) {
            double sum1 = 0;
            for (uchar pixel : cluster1) sum1 += pixel;
            newMean1 = sum1 / cluster1.size();
        }
        
        if (!cluster2.empty()) {
            double sum2 = 0;
            for (uchar pixel : cluster2) sum2 += pixel;
            newMean2 = sum2 / cluster2.size();
        }
        
        // Check convergence
        if (std::abs(newMean1 - mean1) < tolerance && std::abs(newMean2 - mean2) < tolerance) {
            break;
        }
        
        mean1 = newMean1;
        mean2 = newMean2;
    }
    
    // Ensure mean1 is the darker cluster and mean2 is the lighter
    if (mean1 > mean2) {
        std::swap(mean1, mean2);
    }
    
    return std::make_pair(mean1, mean2);
}

cv::Mat CustomThreshold::saturationBasedThreshold(const cv::Mat& input, double intensityWeight, double saturationWeight) {
    cv::Mat hsv, result;
    
    // Convert to HSV color space
    if (input.channels() == 3) {
        cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);
    } else {
        // If grayscale, create a simple threshold
        return fixedThreshold(input, 128.0, 255.0);
    }
    
    result = cv::Mat::zeros(hsv.size(), CV_8UC1);
    
    // Apply saturation-based thresholding
    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(i, j);
            
            double hue = pixel[0] * 2.0;        // Convert from 0-179 to 0-358
            double saturation = pixel[1] / 255.0; // Normalize to 0-1
            double value = pixel[2] / 255.0;      // Normalize to 0-1
            
            // Combine intensity and saturation for thresholding
            // High saturation + low value = likely object
            // Low saturation + high value = likely white background
            double score = intensityWeight * (1.0 - value) + saturationWeight * saturation;
            
            if (score > 0.3) {  // Threshold for object detection
                result.at<uchar>(i, j) = 0;    // Object (black)
            } else {
                result.at<uchar>(i, j) = 255;  // Background (white)
            }
        }
    }
    
    return result;
}

cv::Mat CustomThreshold::preprocessImage(const cv::Mat& input, bool blur, double blurKernel) {
    cv::Mat result = input.clone();
    
    if (blur) {
        int kernelSize = static_cast<int>(blurKernel);
        if (kernelSize % 2 == 0) kernelSize++; // Make sure kernel size is odd
        cv::GaussianBlur(result, result, cv::Size(kernelSize, kernelSize), 0);
    }
    
    return result;
}



std::vector<uchar> CustomThreshold::samplePixels(const cv::Mat& input, int sampleRate) {
    std::vector<uchar> samples;
    
    // Sample every sampleRate-th pixel
    for (int i = 0; i < input.rows; i += sampleRate) {
        for (int j = 0; j < input.cols; j += sampleRate) {
            samples.push_back(input.at<uchar>(i, j));
        }
    }
    
    return samples;
}

cv::Mat CustomThreshold::enhanceSaturation(const cv::Mat& input, double factor) {
    if (input.channels() != 3) {
        return input.clone();
    }
    
    cv::Mat hsv, result;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);
    
    // Enhance saturation channel
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    
    channels[1] *= factor;  // Multiply saturation channel
    
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, result, cv::COLOR_HSV2BGR);
    
    return result;
}

void CustomThreshold::displayHistogram(const cv::Mat& input, const std::string& windowName) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    // Calculate histogram
    std::vector<cv::Mat> images = {gray};
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = range;
    std::vector<int> channels = {0};
    std::vector<int> histSizes = {histSize};
    std::vector<float> ranges = {0, 256};
    
    cv::calcHist(images, channels, cv::Mat(), hist, histSizes, ranges);
    
    // Draw histogram
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
    
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage,
                cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
                cv::Point(bin_w * i, hist_h - cvRound(hist.at<float>(i))),
                cv::Scalar(255, 0, 0), 2, 8, 0);
    }
    
    cv::imshow(windowName, histImage);
}

double CustomThreshold::calculateOtsuThreshold(const cv::Mat& input) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    // Calculate histogram
    std::vector<int> histogram(256, 0);
    int totalPixels = gray.rows * gray.cols;
    
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            histogram[gray.at<uchar>(i, j)]++;
        }
    }
    
    // Calculate cumulative sums and means
    double sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }
    
    double sumB = 0;
    int wB = 0;
    int wF = 0;
    double maxVariance = 0;
    int threshold = 0;
    
    for (int i = 0; i < 256; i++) {
        wB += histogram[i];
        if (wB == 0) continue;
        
        wF = totalPixels - wB;
        if (wF == 0) break;
        
        sumB += i * histogram[i];
        
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;
        
        double betweenVariance = wB * wF * (mB - mF) * (mB - mF);
        
        if (betweenVariance > maxVariance) {
            maxVariance = betweenVariance;
            threshold = i;
        }
    }
    
    return threshold;
}

double CustomThreshold::computeVariance(const std::vector<uchar>& data, double mean) {
    double variance = 0.0;
    for (uchar value : data) {
        variance += (value - mean) * (value - mean);
    }
    return variance / data.size();
}

