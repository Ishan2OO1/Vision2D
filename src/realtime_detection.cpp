#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "custom_threshold.h"
#include "morphological_filter.h"
#include "enhanced_classifier.h"

// Tracked object for temporal smoothing
struct TrackedObject {
    cv::Point2f lastPosition;
    std::string bestClassification;
    double bestConfidence;
    int framesSinceLastSeen;
    int consecutiveDetections;
    double stabilityScore;
    
    TrackedObject() : lastPosition(0, 0), bestClassification("Unknown"), 
                     bestConfidence(0.0), framesSinceLastSeen(0), 
                     consecutiveDetections(0), stabilityScore(0.0) {}
};

// Feature structure for real-time processing
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
    
    // Default constructor
    RegionFeatures() : objectName(""), imageName(""), regionId(0), area(0), percentFilled(0), 
                      aspectRatio(0), compactness(0), majorAxisLength(0), minorAxisLength(0), 
                      orientation(0), eccentricity(0), hu1(0), hu2(0), hu3(0), 
                      centroid(0, 0) {
        boundingRect = cv::RotatedRect();
    }
};

class RealTimeObjectDetector {
private:
    CustomThreshold thresholder;
    MorphologicalFilter morphFilter;
    EnhancedClassifier classifier;
    std::vector<RegionFeatures> trainingDatabase;
    int workingCameraAPI = cv::CAP_ANY;  // Store the working camera API
    
    // Temporal smoothing for stable detection
    std::vector<TrackedObject> trackedObjects;
    cv::Mat previousFrame;
    double stabilizationThreshold = 50.0;  // Distance threshold for tracking
    int minConsecutiveFrames = 3;  // Minimum frames before showing detection

public:
    // Enhanced camera detection optimized for HD webcams like C525
    int findAndSelectCamera() {
        std::cout << "HD Webcam Detection (C525 optimized)..." << std::endl;
        
        // HD webcams like C525 often need different backends - try multiple approaches
        std::vector<int> cameraAPIs = {cv::CAP_DSHOW, cv::CAP_MSMF, cv::CAP_ANY};
        std::vector<std::string> apiNames = {"DirectShow", "Media Foundation", "Auto-detect"};
        
        // Try most common camera indices first (0, 1, 2) with multiple backends
        std::vector<int> commonIndices = {0, 1, 2};
        
        for (int apiIdx = 0; apiIdx < cameraAPIs.size(); apiIdx++) {
            std::cout << "Trying " << apiNames[apiIdx] << " backend..." << std::endl;
            
            for (int i : commonIndices) {
                std::cout << "  Testing camera " << i << " with " << apiNames[apiIdx] << "..." << std::endl;
                cv::VideoCapture cap;
                
                // Try to open camera with specific backend
                bool opened = cap.open(i, cameraAPIs[apiIdx]);
                
                if (opened && cap.isOpened()) {
                    // HD camera optimization - set properties for better performance
                    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);        // Reduce buffer for faster response
                    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);     // Set to moderate resolution
                    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                    cap.set(cv::CAP_PROP_FPS, 30);              // Set desired FPS
                    
                    // For HD webcams, disable autofocus to improve speed
                    cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
                    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
                    
                    // Give HD camera time to stabilize
                    cv::waitKey(300);
                    
                    // Test frame capture
                    cv::Mat testFrame;
                    bool success = cap.read(testFrame);
                    if (success && !testFrame.empty()) {
                        std::cout << "✓ HD Camera " << i << " working with " << apiNames[apiIdx] << "!" << std::endl;
                        
                        // Store the working API for later use
                        workingCameraAPI = cameraAPIs[apiIdx];
                        cap.release();
                        return i;
                    }
                    cap.release();
                }
            }
        }
        
        // Extended search with multiple backends for stubborn HD cameras
        std::cout << "Extended HD search with all backends..." << std::endl;
        for (int apiIdx = 0; apiIdx < cameraAPIs.size(); apiIdx++) {
            std::cout << "Extended search with " << apiNames[apiIdx] << "..." << std::endl;
            for (int i = 3; i < 8; i++) {  // Extended range for HD cameras
                cv::VideoCapture cap;
                if (cap.open(i, cameraAPIs[apiIdx])) {
                    // Apply HD optimizations
                    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
                    cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
                    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                    
                    cv::waitKey(300);  // HD camera stabilization time
                    cv::Mat testFrame;
                    bool success = cap.read(testFrame);
                    if (success && !testFrame.empty()) {
                        std::cout << "✓ HD Camera " << i << " found with " << apiNames[apiIdx] << "!" << std::endl;
                        workingCameraAPI = cameraAPIs[apiIdx];
                        cap.release();
                        return i;
                    }
                    cap.release();
                }
            }
        }
        
        std::cout << "❌ No HD cameras found with any backend!" << std::endl;
        std::cout << "Please check:" << std::endl;
        std::cout << "  • C525 is properly connected via USB" << std::endl;
        std::cout << "  • Camera drivers are installed" << std::endl;
        std::cout << "  • No other applications are using the camera" << std::endl;
        return -1;
    }
    
    // Load training database from CSV
    bool loadTrainingDatabase(const std::string& csvPath) {
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
            
            trainingDatabase.push_back(feature);
        }
        
        file.close();
        std::cout << "Loaded " << trainingDatabase.size() << " training samples from " << csvPath << std::endl;
        
        // Initialize classifier with training data
        classifier.loadTrainingData(trainingDatabase);
        return true;
    }
    
    // Extract features from a single region with enhanced validation
    RegionFeatures extractRegionFeatures(const cv::Mat& binaryImage, int regionId, 
                                       const cv::Mat& labels, const cv::Mat& stats, 
                                       const cv::Mat& centroids) {
        RegionFeatures feature;
        feature.regionId = regionId;
        
        int area = stats.at<int>(regionId, cv::CC_STAT_AREA);
        int left = stats.at<int>(regionId, cv::CC_STAT_LEFT);
        int top = stats.at<int>(regionId, cv::CC_STAT_TOP);
        int width = stats.at<int>(regionId, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(regionId, cv::CC_STAT_HEIGHT);
        
        // Validate region size - skip tiny regions
        if (area < 100 || width < 10 || height < 10) {
            return feature; // Return empty feature
        }
        
        // Create mask for this region
        cv::Mat regionMask = (labels == regionId);
        
        // Calculate moments
        cv::Moments m = cv::moments(regionMask, true);
        if (m.m00 == 0) return feature;
        
        feature.area = m.m00;
        feature.centroid = cv::Point2f(float(m.m10/m.m00), float(m.m01/m.m00));
        
        // Basic geometric features
        double boundingArea = width * height;
        feature.percentFilled = feature.area / boundingArea;
        feature.aspectRatio = double(width) / height;
        
        // Find contour for perimeter calculation
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // Compactness (circularity)
        if (!contours.empty()) {
            double perimeter = cv::arcLength(contours[0], true);
            if (perimeter > 0) {
                feature.compactness = (4 * CV_PI * feature.area) / (perimeter * perimeter);
            }
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
        
        // Hu moments (first 3) with better normalization
        double hu[7];
        cv::HuMoments(m, hu);
        feature.hu1 = hu[0];
        feature.hu2 = hu[1];  
        feature.hu3 = hu[2];
        
        // Validate features - set to defaults if invalid
        if (!std::isfinite(feature.hu1)) feature.hu1 = 0.0;
        if (!std::isfinite(feature.hu2)) feature.hu2 = 0.0;
        if (!std::isfinite(feature.hu3)) feature.hu3 = 0.0;
        if (!std::isfinite(feature.compactness)) feature.compactness = 0.0;
        if (!std::isfinite(feature.eccentricity)) feature.eccentricity = 1.0;
        if (!std::isfinite(feature.orientation)) feature.orientation = 0.0;
        
        // Clamp values to reasonable ranges
        feature.compactness = std::max(0.0, std::min(1.0, feature.compactness));
        feature.percentFilled = std::max(0.0, std::min(1.0, feature.percentFilled));
        feature.eccentricity = std::max(1.0, std::min(10.0, feature.eccentricity));
        
        // Bounding rectangle for visualization
        feature.boundingRect = cv::RotatedRect(feature.centroid, 
                                             cv::Size2f(feature.majorAxisLength, feature.minorAxisLength), 
                                             feature.orientation);
        
        return feature;
    }
    
    // Stabilized object tracking to reduce flickering
    std::vector<RegionFeatures> stabilizeDetections(const std::vector<RegionFeatures>& currentDetections) {
        std::vector<RegionFeatures> stableDetections;
        
        // Update tracking for current detections
        for (const auto& detection : currentDetections) {
            bool found = false;
            double minDistance = stabilizationThreshold;
            int bestMatch = -1;
            
            // Find closest tracked object
            for (int i = 0; i < trackedObjects.size(); i++) {
                double distance = cv::norm(detection.centroid - trackedObjects[i].lastPosition);
                if (distance < minDistance) {
                    minDistance = distance;
                    bestMatch = i;
                    found = true;
                }
            }
            
            if (found && bestMatch >= 0) {
                // Update existing tracked object
                TrackedObject& tracked = trackedObjects[bestMatch];
                tracked.lastPosition = detection.centroid;
                tracked.framesSinceLastSeen = 0;
                tracked.consecutiveDetections++;
                
                // Update classification with temporal smoothing
                if (detection.area > 0) {  // Valid detection
                    auto result = classifier.classifyMultiple(detection);
                    if (!result.empty() && result[0].second > 0.15) {
                        std::string newClass = result[0].first;
                        double newConf = result[0].second;
                        
                        // Accept all classifications with reasonable confidence
                        // Remove restrictive validation that might be causing issues
                        if (newConf > 0.15) {
                            // Smooth confidence over time
                            if (newClass == tracked.bestClassification) {
                                tracked.bestConfidence = 0.7 * tracked.bestConfidence + 0.3 * newConf;
                                tracked.stabilityScore += 0.1;
                            } else if (newConf > tracked.bestConfidence + 0.1) {
                                tracked.bestClassification = newClass;
                                tracked.bestConfidence = newConf;
                                tracked.stabilityScore = 0.0;  // Reset stability
                            }
                        }
                    }
                }
                
                // Add to stable detections if confident enough
                if (tracked.consecutiveDetections >= minConsecutiveFrames && 
                    tracked.bestConfidence > 0.2 && tracked.stabilityScore > 0.3) {
                    
                    RegionFeatures stableFeature = detection;
                    stableDetections.push_back(stableFeature);
                }
            } else {
                // New object detected
                if (detection.area > 300) {  // Only track substantial objects
                    TrackedObject newTracked;
                    newTracked.lastPosition = detection.centroid;
                    newTracked.framesSinceLastSeen = 0;
                    newTracked.consecutiveDetections = 1;
                    
                    // Initial classification
                    auto result = classifier.classifyMultiple(detection);
                    if (!result.empty() && result[0].second > 0.15) {
                        std::string newClass = result[0].first;
                        double newConf = result[0].second;
                        
                        // Accept all valid classifications
                        if (newConf > 0.15) {
                            newTracked.bestClassification = newClass;
                            newTracked.bestConfidence = newConf;
                        }
                    }
                    
                    trackedObjects.push_back(newTracked);
                }
            }
        }
        
        // Age out old tracked objects
        for (int i = trackedObjects.size() - 1; i >= 0; i--) {
            trackedObjects[i].framesSinceLastSeen++;
            if (trackedObjects[i].framesSinceLastSeen > 10) {  // Remove after 10 frames
                trackedObjects.erase(trackedObjects.begin() + i);
            }
        }
        
        return stableDetections;
    }

    // Visualize detection results with stability info
    void visualizeDetection(cv::Mat& frame, const RegionFeatures& feature, 
                          const std::string& classification, double confidence) {
        // Draw oriented bounding box
        cv::Point2f vertices[4];
        feature.boundingRect.points(vertices);
        
        for (int i = 0; i < 4; i++) {
            cv::line(frame, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 0), 2);
        }
        
        // Draw major and minor axes
        float cos_angle = cos(feature.orientation * CV_PI / 180.0);
        float sin_angle = sin(feature.orientation * CV_PI / 180.0);
        
        // Major axis
        cv::Point2f major_end1 = feature.centroid + 
            cv::Point2f(cos_angle * feature.majorAxisLength/2, sin_angle * feature.majorAxisLength/2);
        cv::Point2f major_end2 = feature.centroid - 
            cv::Point2f(cos_angle * feature.majorAxisLength/2, sin_angle * feature.majorAxisLength/2);
        cv::line(frame, major_end1, major_end2, cv::Scalar(255, 0, 0), 2);
        
        // Minor axis
        cv::Point2f minor_end1 = feature.centroid + 
            cv::Point2f(-sin_angle * feature.minorAxisLength/2, cos_angle * feature.minorAxisLength/2);
        cv::Point2f minor_end2 = feature.centroid - 
            cv::Point2f(-sin_angle * feature.minorAxisLength/2, cos_angle * feature.minorAxisLength/2);
        cv::line(frame, minor_end1, minor_end2, cv::Scalar(0, 0, 255), 2);
        
        // Draw centroid
        cv::circle(frame, feature.centroid, 5, cv::Scalar(255, 255, 0), -1);
        
        // Draw classification label and confidence
        std::string label = classification + " (" + std::to_string(int(confidence * 100)) + "%)";
        cv::putText(frame, label, cv::Point(feature.centroid.x - 50, feature.centroid.y - 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Draw feature info
        std::string info = "Area: " + std::to_string(int(feature.area)) + 
                          " AR: " + std::to_string(feature.aspectRatio).substr(0, 4);
        cv::putText(frame, info, cv::Point(feature.centroid.x - 50, feature.centroid.y + 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    }
    
public:
    RealTimeObjectDetector() {
        // Load training database
        if (!loadTrainingDatabase("results/comprehensive_feature_vectors.csv")) {
            std::cerr << "Warning: Could not load training database. Classification disabled." << std::endl;
        }
    }
    
    void run() {
        // Find and select camera
        int cameraIndex = findAndSelectCamera();
        if (cameraIndex < 0) {
            std::cout << "No camera available. Please ensure a camera is connected." << std::endl;
            return;
        }
        
        std::cout << "\nInitializing HD camera " << cameraIndex << " with optimal backend..." << std::endl;
        cv::VideoCapture cap;
        
        // Use the working API that was detected during camera search
        bool opened = cap.open(cameraIndex, workingCameraAPI);
        
        if (!opened || !cap.isOpened()) {
            std::cerr << "Error: Could not open HD camera " << cameraIndex << std::endl;
            // Try alternative initialization methods for HD cameras
            std::cout << "Trying alternative HD camera backends..." << std::endl;
            
            // Try Media Foundation
            if (cap.open(cameraIndex, cv::CAP_MSMF)) {
                std::cout << "✓ Opened with Media Foundation" << std::endl;
            }
            // Try DirectShow
            else if (cap.open(cameraIndex, cv::CAP_DSHOW)) {
                std::cout << "✓ Opened with DirectShow" << std::endl;  
            }
            // Try default
            else if (cap.open(cameraIndex)) {
                std::cout << "✓ Opened with default backend" << std::endl;
            }
            else {
                std::cerr << "Error: HD camera initialization failed with all backends" << std::endl;
                return;
            }
        }
        else {
            std::cout << "✓ Opened HD camera with detected backend" << std::endl;
        }
        
        // HD Camera optimized properties - set in optimal order
        std::cout << "Configuring HD camera properties..." << std::endl;
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);         // Critical: Reduce buffer first
        cap.set(cv::CAP_PROP_AUTOFOCUS, 0);          // Disable autofocus for speed
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);      // Fast auto exposure
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);      // Set resolution
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);               // Set FPS
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G')); // Use MJPEG for faster streaming
        
        // Test frame capture with HD camera optimization
        cv::Mat testFrame;
        // Give HD camera minimal time to stabilize settings
        cv::waitKey(300);  // Reduced wait time for HD cameras
        
        // Try multiple frame captures to ensure camera is ready
        bool frameSuccess = false;
        for (int attempt = 0; attempt < 3; attempt++) {
            cap >> testFrame;
            if (!testFrame.empty()) {
                frameSuccess = true;
                break;
            }
            cv::waitKey(100);  // Brief wait between attempts
        }
        
        if (!frameSuccess) {
            std::cerr << "Error: Cannot capture frames from HD camera" << std::endl;
            return;
        }
        
        // Get actual camera properties
        int actualWidth = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int actualHeight = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double actualFPS = cap.get(cv::CAP_PROP_FPS);
        
        std::cout << "✓ Camera successfully initialized!" << std::endl;
        std::cout << "  Resolution: " << actualWidth << "x" << actualHeight << std::endl;
        std::cout << "  FPS: " << actualFPS << std::endl;
        
        // Display windows
        cv::namedWindow("Real-time Object Detection", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Thresholded", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Cleaned Binary", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Region Map", cv::WINDOW_AUTOSIZE);
        
        // Processing parameters
        bool captureMode = false;
        std::string collectingLabel = "";
        std::vector<RegionFeatures> capturedData;
        
        // FPS calculation
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        double fps = 0.0;
        
        std::cout << "\n=== REAL-TIME OBJECT DETECTION CONTROLS ===" << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│ DETECTION CONTROLS:                                 │" << std::endl;
        std::cout << "│  ESC/Q: Quit application                           │" << std::endl;
        std::cout << "│  SPACE: Toggle processing windows display          │" << std::endl;
        std::cout << "│                                                     │" << std::endl;
        std::cout << "│ TASK 5 - DATA COLLECTION CONTROLS:                 │" << std::endl;
        std::cout << "│  C: Capture snapshot for training (enter name)     │" << std::endl;
        std::cout << "│  S: Save captured data to CSV                      │" << std::endl;
        std::cout << "│  R: Reset and clear captured data                  │" << std::endl;
        std::cout << "└─────────────────────────────────────────────────────┘" << std::endl;
        
        cv::Mat frame, thresholded, cleaned, regionMap;
        bool showProcessing = true;
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            // FPS calculation
            frameCount++;
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);
            if (duration.count() > 1000) {
                fps = frameCount * 1000.0 / duration.count();
                frameCount = 0;
                lastTime = currentTime;
            }
            
            // Enhanced frame processing with temporal smoothing
            cv::Mat preprocessed = thresholder.preprocessImage(frame, true, 7); // Extra blur for stability
            
            // Temporal smoothing with previous frame
            if (!previousFrame.empty() && previousFrame.size() == preprocessed.size()) {
                cv::Mat smoothed;
                cv::addWeighted(preprocessed, 0.7, previousFrame, 0.3, 0, smoothed);
                preprocessed = smoothed;
            }
            previousFrame = preprocessed.clone();
            
            thresholded = thresholder.kmeansThreshold(preprocessed, 24); // Fewer clusters for more stable regions
            
            // Multi-stage morphological filtering for better object separation
            cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            
            // First pass: remove small noise
            cv::morphologyEx(thresholded, cleaned, cv::MORPH_OPEN, kernel1);
            // Second pass: fill holes
            cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel2);
            // Third pass: smooth boundaries
            cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel1);
            
            // Connected components analysis
            cv::Mat labels, stats, centroids;
            int numComponents = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids, 8, CV_32S);
            
            // Create region visualization
            regionMap = cv::Mat::zeros(cleaned.size(), CV_8UC3);
            std::vector<cv::Vec3b> colors(numComponents);
            for (int i = 1; i < numComponents; i++) {
                colors[i] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
            }
            
            // Process each region - collect all potential detections first
            std::vector<RegionFeatures> currentDetections;
            for (int i = 1; i < numComponents; i++) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                int left = stats.at<int>(i, cv::CC_STAT_LEFT);
                int top = stats.at<int>(i, cv::CC_STAT_TOP);
                int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
                int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                
                // Filter regions - focus on substantial objects for bottles/cans
                bool touchesBoundary = (left <= 1) || (top <= 1) || 
                                     (left + width >= frame.cols - 1) || 
                                     (top + height >= frame.rows - 1);
                
                // Stricter area filtering for more stable detection
                if (area > 300 && (!touchesBoundary || area > 1500)) {
                    // Extract features
                    RegionFeatures feature = extractRegionFeatures(cleaned, i, labels, stats, centroids);
                    
                    if (feature.area > 0 && feature.compactness > 0.1) {  // Basic validity check
                        currentDetections.push_back(feature);
                        
                        // Store data if in capture mode (will be captured on C press)
                        if (captureMode && !collectingLabel.empty()) {
                            feature.objectName = collectingLabel;
                            feature.imageName = "snapshot_" + std::to_string(capturedData.size());
                            // Don't add yet - will capture on next 'C' press
                        }
                    }
                }
            }
            
            // Apply temporal stabilization to reduce flickering
            std::vector<RegionFeatures> stableDetections = stabilizeDetections(currentDetections);
            
            // Visualize only stable detections
            std::vector<RegionFeatures> detectedObjects;
            for (const auto& feature : stableDetections) {
                // Find the corresponding tracked object for classification
                std::string classification = "Unknown";
                double confidence = 0.0;
                
                for (const auto& tracked : trackedObjects) {
                    double distance = cv::norm(feature.centroid - tracked.lastPosition);
                    if (distance < stabilizationThreshold) {
                        classification = tracked.bestClassification;
                        confidence = tracked.bestConfidence;
                        break;
                    }
                }
                
                // Display all valid classifications
                if (classification != "Unknown" && confidence > 0.2) {
                    visualizeDetection(frame, feature, classification, confidence);
                    detectedObjects.push_back(feature);
                }
            }
            
            // Color regions in region map for stable detections only
            for (const auto& feature : detectedObjects) {
                // Find the corresponding region ID
                for (int i = 1; i < numComponents; i++) {
                    cv::Point2f regionCenter(centroids.at<double>(i, 0), centroids.at<double>(i, 1));
                    double distance = cv::norm(feature.centroid - regionCenter);
                    if (distance < 10.0) {  // Close enough match
                        for (int y = 0; y < regionMap.rows; y++) {
                            for (int x = 0; x < regionMap.cols; x++) {
                                if (labels.at<int>(y, x) == i) {
                                    regionMap.at<cv::Vec3b>(y, x) = colors[i];
                                }
                            }
                        }
                        break;
                    }
                }
            }
            
            // Display FPS and statistics
            std::string fpsText = "FPS: " + std::to_string(fps).substr(0, 5);
            cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            std::string objectCount = "Objects: " + std::to_string(detectedObjects.size());
            cv::putText(frame, objectCount, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            if (captureMode) {
                // Capture mode status display
                cv::Scalar textColor = cv::Scalar(0, 255, 255); // Yellow for capture ready
                std::string captureText = "CAPTURE READY: " + collectingLabel;
                cv::putText(frame, captureText, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, textColor, 2);
                
                std::string instruction = "Press C to capture current frame";
                cv::putText(frame, instruction, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);
                
                std::string instructions = "Press 'S' to save, 'R' to reset";
                cv::putText(frame, instructions, cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);
                
                // Draw capture indicator border
                cv::rectangle(frame, cv::Point(5, 70), cv::Point(frame.cols-5, 170), textColor, 3);
            }
            
            if (!capturedData.empty()) {
                // Show captured data status
                cv::Scalar textColor = cv::Scalar(0, 255, 0); // Green for captured
                std::string capturedText = "CAPTURED: " + std::to_string(capturedData.size()) + " regions";
                cv::putText(frame, capturedText, cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);
            }
            
            // Display results
            cv::imshow("Real-time Object Detection", frame);
            
            if (showProcessing) {
                cv::imshow("Thresholded", thresholded);
                cv::imshow("Cleaned Binary", cleaned);
                cv::imshow("Region Map", regionMap);
            }
            
            // Handle key presses
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') break; // ESC or Q
            
            if (key == 'c' || key == 'C') {
                if (!captureMode) {
                    // First C press - enter capture mode and get object name
                    std::cout << "\n*** TASK 5 SNAPSHOT CAPTURE ***" << std::endl;
                    std::cout << "Enter object name for capture: ";
                    std::cin.ignore(); // Clear input buffer
                    std::getline(std::cin, collectingLabel);
                    if (!collectingLabel.empty()) {
                        captureMode = true;
                        capturedData.clear();
                        std::cout << "Capture mode ready for: " << collectingLabel << std::endl;
                        std::cout << "Position the object and press 'C' again to capture snapshot." << std::endl;
                        std::cout << "Press 'S' to save captured data, 'R' to reset." << std::endl;
                    }
                } else {
                    // Second C press - capture current frame
                    std::cout << "\n*** CAPTURING SNAPSHOT ***" << std::endl;
                    
                    // Capture all current detections
                    for (const auto& feature : currentDetections) {
                        RegionFeatures captureFeature = feature;
                        captureFeature.objectName = collectingLabel;
                        captureFeature.imageName = "snapshot_" + std::to_string(capturedData.size());
                        capturedData.push_back(captureFeature);
                    }
                    
                    std::cout << "✓ Captured " << currentDetections.size() << " regions for " << collectingLabel << std::endl;
                    std::cout << "Total captured: " << capturedData.size() << " regions" << std::endl;
                    std::cout << "Press 'C' to capture another frame, 'S' to save, or 'R' to reset." << std::endl;
                }
            }
            
            if (key == 's' || key == 'S') {
                if (captureMode && !capturedData.empty()) {
                    std::cout << "\n*** TASK 5 DATA SAVING ***" << std::endl;
                    std::cout << "Saving " << capturedData.size() << " captured regions for " << collectingLabel << "..." << std::endl;
                    
                    // Save new training data to CSV
                    std::ofstream csvFile("results/realtime_training_data.csv", std::ios::app);
                    if (csvFile.is_open()) {
                        // Write header if file is empty
                        csvFile.seekp(0, std::ios::end);
                        if (csvFile.tellp() == 0) {
                            csvFile << "Object,ImageName,RegionId,Area,PercentFilled,AspectRatio,Compactness,"
                                   << "MajorAxisLength,MinorAxisLength,Orientation,Eccentricity,Hu1,Hu2,Hu3" << std::endl;
                        }
                        
                        for (const auto& feature : capturedData) {
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
                        std::cout << "✓ Successfully saved " << capturedData.size() 
                                 << " feature vectors for " << collectingLabel << std::endl;
                        std::cout << "  Saved to: results/realtime_training_data.csv" << std::endl;
                        
                        // Also append to comprehensive training database
                        std::ofstream mainCsv("results/comprehensive_feature_vectors.csv", std::ios::app);
                        if (mainCsv.is_open()) {
                            for (const auto& feature : capturedData) {
                                mainCsv << feature.objectName << "," 
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
                            mainCsv.close();
                            std::cout << "✓ Also appended to main training database" << std::endl;
                        }
                    } else {
                        std::cerr << "Error: Could not save training data" << std::endl;
                    }
                } else if (captureMode) {
                    std::cout << "No data captured yet for " << collectingLabel << std::endl;
                } else {
                    std::cout << "Not in capture mode. Press 'C' to start." << std::endl;
                }
                captureMode = false;
                collectingLabel = "";
                capturedData.clear();
            }
            
            if (key == 'r' || key == 'R') {
                if (captureMode || !capturedData.empty()) {
                    std::cout << "\n*** TASK 5 CAPTURE RESET ***" << std::endl;
                    std::cout << "Cleared " << capturedData.size() << " captured regions for " << collectingLabel << std::endl;
                } else {
                    std::cout << "No active capture to reset" << std::endl;
                }
                captureMode = false;
                collectingLabel = "";
                capturedData.clear();
            }
            
            if (key == ' ') {
                showProcessing = !showProcessing;
                if (!showProcessing) {
                    cv::destroyWindow("Thresholded");
                    cv::destroyWindow("Cleaned Binary");
                    cv::destroyWindow("Region Map");
                }
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
        
        std::cout << "Real-time detection stopped." << std::endl;
    }
};

int main() {
    std::cout << "=== REAL-TIME OBJECT DETECTION SYSTEM ===" << std::endl;
    std::cout << "Translation, Scale, and Rotation Invariant" << std::endl;
    std::cout << "Tasks 1-6 Integration with Live Camera Feed" << std::endl;
    
    RealTimeObjectDetector detector;
    detector.run();
    
    return 0;
}