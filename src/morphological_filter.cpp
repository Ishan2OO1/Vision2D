#include "morphological_filter.h"
#include <iostream>
#include <algorithm>
#include <queue>

MorphologicalFilter::MorphologicalFilter() {
    // Constructor implementation
}

cv::Mat MorphologicalFilter::erosion(const cv::Mat& input, const cv::Mat& kernel) {
    // Ensure input is binary
    cv::Mat binary;
    if (input.channels() == 3) {
        cv::cvtColor(input, binary, cv::COLOR_BGR2GRAY);
    } else {
        binary = input.clone();
    }
    
    cv::Mat result = cv::Mat::zeros(binary.size(), CV_8UC1);
    
    int kRows = kernel.rows;
    int kCols = kernel.cols;
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;
    
    // Erosion: Output pixel is 1 (white) only if ALL kernel pixels fit in foreground
    for (int i = kCenterY; i < binary.rows - kCenterY; i++) {
        for (int j = kCenterX; j < binary.cols - kCenterX; j++) {
            bool allFit = true;
            
            // Check if all kernel positions have foreground pixels
            for (int ki = 0; ki < kRows && allFit; ki++) {
                for (int kj = 0; kj < kCols && allFit; kj++) {
                    if (kernel.at<uchar>(ki, kj) > 0) { // If kernel element is active
                        int imageX = j + kj - kCenterX;
                        int imageY = i + ki - kCenterY;
                        
                        if (imageX >= 0 && imageX < binary.cols && 
                            imageY >= 0 && imageY < binary.rows) {
                            if (binary.at<uchar>(imageY, imageX) == 0) { // Background pixel
                                allFit = false;
                            }
                        } else {
                            allFit = false; // Outside image boundary
                        }
                    }
                }
            }
            
            result.at<uchar>(i, j) = allFit ? 255 : 0;
        }
    }
    
    return result;
}

cv::Mat MorphologicalFilter::dilation(const cv::Mat& input, const cv::Mat& kernel) {
    // Ensure input is binary
    cv::Mat binary;
    if (input.channels() == 3) {
        cv::cvtColor(input, binary, cv::COLOR_BGR2GRAY);
    } else {
        binary = input.clone();
    }
    
    cv::Mat result = cv::Mat::zeros(binary.size(), CV_8UC1);
    
    int kRows = kernel.rows;
    int kCols = kernel.cols;
    int kCenterX = kCols / 2;
    int kCenterY = kRows / 2;
    
    // Dilation: Output pixel is 1 (white) if ANY kernel pixel hits foreground
    for (int i = kCenterY; i < binary.rows - kCenterY; i++) {
        for (int j = kCenterX; j < binary.cols - kCenterX; j++) {
            bool anyHit = false;
            
            // Check if any kernel position has foreground pixel
            for (int ki = 0; ki < kRows && !anyHit; ki++) {
                for (int kj = 0; kj < kCols && !anyHit; kj++) {
                    if (kernel.at<uchar>(ki, kj) > 0) { // If kernel element is active
                        int imageX = j + kj - kCenterX;
                        int imageY = i + ki - kCenterY;
                        
                        if (imageX >= 0 && imageX < binary.cols && 
                            imageY >= 0 && imageY < binary.rows) {
                            if (binary.at<uchar>(imageY, imageX) > 0) { // Foreground pixel
                                anyHit = true;
                            }
                        }
                    }
                }
            }
            
            result.at<uchar>(i, j) = anyHit ? 255 : 0;
        }
    }
    
    return result;
}

cv::Mat MorphologicalFilter::opening(const cv::Mat& input, const cv::Mat& kernel) {
    // Opening = Erosion followed by Dilation
    // Removes noise while preserving object shape
    cv::Mat eroded = erosion(input, kernel);
    cv::Mat opened = dilation(eroded, kernel);
    return opened;
}

cv::Mat MorphologicalFilter::closing(const cv::Mat& input, const cv::Mat& kernel) {
    // Closing = Dilation followed by Erosion  
    // Fills holes and gaps while preserving object shape
    cv::Mat dilated = dilation(input, kernel);
    cv::Mat closed = erosion(dilated, kernel);
    return closed;
}

cv::Mat MorphologicalFilter::morphologicalGradient(const cv::Mat& input, const cv::Mat& kernel) {
    // Gradient = Dilation - Erosion
    // Highlights object boundaries
    cv::Mat dilated = dilation(input, kernel);
    cv::Mat eroded = erosion(input, kernel);
    
    cv::Mat gradient;
    cv::subtract(dilated, eroded, gradient);
    return gradient;
}

cv::Mat MorphologicalFilter::topHat(const cv::Mat& input, const cv::Mat& kernel) {
    // Top Hat = Original - Opening
    // Highlights small bright details
    cv::Mat opened = opening(input, kernel);
    cv::Mat tophat;
    cv::subtract(input, opened, tophat);
    return tophat;
}

cv::Mat MorphologicalFilter::blackHat(const cv::Mat& input, const cv::Mat& kernel) {
    // Black Hat = Closing - Original
    // Highlights small dark details  
    cv::Mat closed = closing(input, kernel);
    cv::Mat blackhat;
    cv::subtract(closed, input, blackhat);
    return blackhat;
}

cv::Mat MorphologicalFilter::createKernel(StructureShape shape, int size) {
    return createKernel(size, size, shape);
}

cv::Mat MorphologicalFilter::createKernel(int width, int height, StructureShape shape) {
    cv::Mat kernel = cv::Mat::zeros(height, width, CV_8UC1);
    
    int centerX = width / 2;
    int centerY = height / 2;
    
    switch (shape) {
        case RECTANGULAR:
            kernel = cv::Mat::ones(height, width, CV_8UC1) * 255;
            break;
            
        case ELLIPTICAL:
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    double dx = (j - centerX) / (double)centerX;
                    double dy = (i - centerY) / (double)centerY;
                    if (dx*dx + dy*dy <= 1.0) {
                        kernel.at<uchar>(i, j) = 255;
                    }
                }
            }
            break;
            
        case CROSS:
            // Horizontal line
            for (int j = 0; j < width; j++) {
                kernel.at<uchar>(centerY, j) = 255;
            }
            // Vertical line
            for (int i = 0; i < height; i++) {
                kernel.at<uchar>(i, centerX) = 255;
            }
            break;
            
        default:
            kernel = cv::Mat::ones(height, width, CV_8UC1) * 255;
            break;
    }
    
    return kernel;
}

cv::Mat MorphologicalFilter::removeNoise(const cv::Mat& input, int kernelSize) {
    // Strategy: Opening operation to remove small noise blobs
    cv::Mat kernel = createKernel(ELLIPTICAL, kernelSize);
    cv::Mat result = opening(input, kernel);
    
    std::cout << "Noise removal: Opening with " << kernelSize << "x" << kernelSize << " elliptical kernel" << std::endl;
    return result;
}

cv::Mat MorphologicalFilter::fillHoles(const cv::Mat& input, int kernelSize) {
    // Strategy: Closing operation to fill holes in objects
    cv::Mat kernel = createKernel(ELLIPTICAL, kernelSize);
    cv::Mat result = closing(input, kernel);
    
    std::cout << "Hole filling: Closing with " << kernelSize << "x" << kernelSize << " elliptical kernel" << std::endl;
    return result;
}

cv::Mat MorphologicalFilter::smoothBoundaries(const cv::Mat& input, int kernelSize) {
    // Strategy: Opening followed by closing to smooth boundaries
    cv::Mat kernel = createKernel(ELLIPTICAL, kernelSize);
    cv::Mat opened = opening(input, kernel);
    cv::Mat result = closing(opened, kernel);
    
    std::cout << "Boundary smoothing: Opening + Closing with " << kernelSize << "x" << kernelSize << " elliptical kernel" << std::endl;
    return result;
}

cv::Mat MorphologicalFilter::cleanupObjects(const cv::Mat& input, int noiseSize, int holeSize) {
    // Comprehensive cleanup strategy
    std::cout << "Comprehensive cleanup strategy:" << std::endl;
    
    // Step 1: Remove small noise (opening)
    cv::Mat step1 = removeNoise(input, noiseSize);
    
    // Step 2: Fill holes in objects (closing)  
    cv::Mat step2 = fillHoles(step1, holeSize);
    
    // Step 3: Final smoothing
    cv::Mat result = smoothBoundaries(step2, 3);
    
    return result;
}

void MorphologicalFilter::analyzeImage(const cv::Mat& input, const std::string& imageName) {
    std::cout << "\\n=== Morphological Analysis: " << imageName << " ===" << std::endl;
    
    // Basic statistics
    int totalPixels = input.rows * input.cols;
    int foregroundPixels = cv::countNonZero(input);
    int backgroundPixels = totalPixels - foregroundPixels;
    
    double foregroundRatio = (double)foregroundPixels / totalPixels;
    
    std::cout << "Image size: " << input.cols << "x" << input.rows << " (" << totalPixels << " pixels)" << std::endl;
    std::cout << "Foreground pixels: " << foregroundPixels << " (" << (foregroundRatio * 100) << "%)" << std::endl;
    std::cout << "Background pixels: " << backgroundPixels << " (" << ((1-foregroundRatio) * 100) << "%)" << std::endl;
    
    // Object count estimation
    int objectCount = countObjects(input);
    std::cout << "Estimated object regions: " << objectCount << std::endl;
    
    // Noise ratio estimation
    double noiseRatio = calculateNoiseRatio(input);
    std::cout << "Estimated noise ratio: " << (noiseRatio * 100) << "%" << std::endl;
    
    // Recommendations
    std::cout << "Recommended cleanup strategy:" << std::endl;
    if (noiseRatio > 0.1) {
        std::cout << "  - High noise detected: Use opening (3x3 kernel) to remove small blobs" << std::endl;
    }
    if (foregroundRatio < 0.3 && objectCount > 1) {
        std::cout << "  - Multiple objects with potential holes: Use closing (5x5 kernel)" << std::endl;
    }
    if (objectCount > 5) {
        std::cout << "  - Many small regions: Consider larger opening kernel (5x5)" << std::endl;
    }
    std::cout << "===========================================" << std::endl;
}

int MorphologicalFilter::countObjects(const cv::Mat& binaryImage) {
    // Simple connected component counting using flood fill
    cv::Mat temp = binaryImage.clone();
    int objectCount = 0;
    
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            if (temp.at<uchar>(i, j) > 0) {
                // Found new object, flood fill it
                cv::floodFill(temp, cv::Point(j, i), cv::Scalar(0));
                objectCount++;
            }
        }
    }
    
    return objectCount;
}

double MorphologicalFilter::calculateNoiseRatio(const cv::Mat& binaryImage) {
    // Estimate noise by counting very small connected components
    cv::Mat temp = binaryImage.clone();
    int totalObjects = 0;
    int smallObjects = 0;
    
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            if (temp.at<uchar>(i, j) > 0) {
                // Count pixels in this component
                cv::Mat mask = cv::Mat::zeros(temp.rows + 2, temp.cols + 2, CV_8UC1);
                int area = cv::floodFill(temp, mask, cv::Point(j, i), cv::Scalar(0));
                
                totalObjects++;
                if (area < 50) { // Objects smaller than 50 pixels considered noise
                    smallObjects++;
                }
            }
        }
    }
    
    return totalObjects > 0 ? (double)smallObjects / totalObjects : 0.0;
}