#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <filesystem>
#include "custom_threshold.h"
#include "morphological_filter.h"
#include "connected_components.h"

// Region Features structure (consistent with task4)
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
    
    // Constructor
    RegionFeatures() : objectName(""), imageName(""), regionId(0), area(0), percentFilled(0), 
                      aspectRatio(0), compactness(0), majorAxisLength(0), minorAxisLength(0), 
                      orientation(0), eccentricity(0), hu1(0), hu2(0), hu3(0), 
                      centroid(0, 0), boundingRect() {}
};

class HybridDataCollector {
private:
    CustomThreshold thresholder;
    MorphologicalFilter morphFilter;
    std::vector<RegionFeatures> existingDatabase;
    std::vector<RegionFeatures> newCollectedData;
    
    // Camera detection and selection
    int findAndSelectCamera() {
        std::cout << "Searching for available cameras..." << std::endl;
        std::vector<int> availableCameras;
        
        // Scan for available cameras
        for (int i = 0; i < 10; i++) {
            cv::VideoCapture cap(i);
            if (cap.isOpened()) {
                cv::Mat testFrame;
                cap >> testFrame;
                if (!testFrame.empty()) {
                    std::cout << "Camera " << i << " detected and working" << std::endl;
                    availableCameras.push_back(i);
                }
                cap.release();
            }
        }
        
        if (availableCameras.empty()) {
            std::cout << "No working cameras found!" << std::endl;
            return -1;
        }
        
        // If only one camera, use it
        if (availableCameras.size() == 1) {
            std::cout << "Only one camera found. Using camera " << availableCameras[0] << std::endl;
            return availableCameras[0];
        }
        
        // Multiple cameras - ask user to choose
        std::cout << "\nMultiple cameras detected:" << std::endl;
        for (size_t i = 0; i < availableCameras.size(); i++) {
            std::cout << "  " << (i + 1) << ". Camera " << availableCameras[i] << std::endl;
        }
        
        int choice = -1;
        while (choice < 1 || choice > (int)availableCameras.size()) {
            std::cout << "\nSelect camera (1-" << availableCameras.size() << "): ";
            std::cin >> choice;
            
            if (std::cin.fail()) {
                std::cin.clear();
                std::cin.ignore(10000, '\n');
                choice = -1;
            }
        }
        
        int selectedCamera = availableCameras[choice - 1];
        std::cout << "Selected camera " << selectedCamera << std::endl;
        return selectedCamera;
    }
    
    // Load existing training database
    bool loadExistingDatabase(const std::string& csvPath) {
        existingDatabase.clear();
        std::ifstream file(csvPath);
        
        if (!file.is_open()) {
            std::cout << "No existing database found at " << csvPath << ". Starting fresh." << std::endl;
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
            
            existingDatabase.push_back(feature);
        }
        
        file.close();
        std::cout << "Loaded " << existingDatabase.size() << " existing training samples" << std::endl;
        return true;
    }
    
    // Extract features from a region
    RegionFeatures extractRegionFeatures(const cv::Mat& binaryImage, int regionId, 
                                       const cv::Mat& labels, const cv::Mat& stats, 
                                       const cv::Mat& centroids, const std::string& objectName, 
                                       const std::string& imageName) {
        RegionFeatures feature;
        feature.objectName = objectName;
        feature.imageName = imageName;
        feature.regionId = regionId;
        
        int area = stats.at<int>(regionId, cv::CC_STAT_AREA);
        int left = stats.at<int>(regionId, cv::CC_STAT_LEFT);
        int top = stats.at<int>(regionId, cv::CC_STAT_TOP);
        int width = stats.at<int>(regionId, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(regionId, cv::CC_STAT_HEIGHT);
        
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
        
        // Hu moments (first 3)
        double hu[7];
        cv::HuMoments(m, hu);
        feature.hu1 = hu[0];
        feature.hu2 = hu[1];
        feature.hu3 = hu[2];
        
        // Bounding rectangle
        feature.boundingRect = cv::RotatedRect(feature.centroid, 
                                             cv::Size2f(feature.majorAxisLength, feature.minorAxisLength), 
                                             feature.orientation);
        
        return feature;
    }
    
    // Visualize detected objects during collection
    void visualizeDetection(cv::Mat& frame, const RegionFeatures& feature, bool isNew = true) {
        // Choose color based on whether it's a new sample
        cv::Scalar color = isNew ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 0);
        
        // Draw oriented bounding box
        cv::Point2f vertices[4];
        feature.boundingRect.points(vertices);
        
        for (int i = 0; i < 4; i++) {
            cv::line(frame, vertices[i], vertices[(i+1)%4], color, 2);
        }
        
        // Draw centroid
        cv::circle(frame, feature.centroid, 5, color, -1);
        
        // Draw object name
        cv::putText(frame, feature.objectName, 
                   cv::Point(feature.centroid.x - 50, feature.centroid.y - 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        
        // Draw area info
        std::string areaText = "A: " + std::to_string(int(feature.area));
        cv::putText(frame, areaText, 
                   cv::Point(feature.centroid.x - 30, feature.centroid.y + 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
    
    // Save hybrid database (existing + new data)
    void saveHybridDatabase(const std::string& outputPath) {
        std::ofstream csvFile(outputPath);
        if (!csvFile.is_open()) {
            std::cerr << "Could not create output file: " << outputPath << std::endl;
            return;
        }
        
        // Write header
        csvFile << "Object,ImageName,RegionId,Area,PercentFilled,AspectRatio,Compactness,"
                << "MajorAxisLength,MinorAxisLength,Orientation,Eccentricity,Hu1,Hu2,Hu3" << std::endl;
        
        // Write existing data
        for (const auto& feature : existingDatabase) {
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
        
        // Write new collected data
        for (const auto& feature : newCollectedData) {
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
        
        std::cout << "\n=== HYBRID DATABASE SAVED ===" << std::endl;
        std::cout << "Existing samples: " << existingDatabase.size() << std::endl;
        std::cout << "New samples: " << newCollectedData.size() << std::endl;
        std::cout << "Total samples: " << (existingDatabase.size() + newCollectedData.size()) << std::endl;
        std::cout << "Saved to: " << outputPath << std::endl;
    }
    
public:
    HybridDataCollector() {
        // Load existing database from task4
        loadExistingDatabase("results/comprehensive_feature_vectors.csv");
    }
    
    void run() {
        // Find and select camera
        int cameraIndex = findAndSelectCamera();
        if (cameraIndex < 0) {
            std::cerr << "No camera available for data collection!" << std::endl;
            return;
        }
        
        std::cout << "\nInitializing camera " << cameraIndex << " for data collection..." << std::endl;
        cv::VideoCapture cap(cameraIndex);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera " << cameraIndex << std::endl;
            return;
        }
        
        // Get actual camera properties
        int actualWidth = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int actualHeight = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double actualFPS = cap.get(cv::CAP_PROP_FPS);
        
        std::cout << "Camera configured: " << actualWidth << "x" << actualHeight 
                  << " @ " << actualFPS << " FPS" << std::endl;
        
        // Display windows
        cv::namedWindow("Hybrid Data Collection", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Binary Regions", cv::WINDOW_AUTOSIZE);
        
        // Collection state
        bool collectingData = false;
        std::string currentObjectName = "";
        int sampleCount = 0;
        
        // FPS calculation
        auto lastTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        double fps = 0.0;
        
        std::cout << "\n=== HYBRID DATA COLLECTION SYSTEM ===" << std::endl;
        std::cout << "Existing database: " << existingDatabase.size() << " samples loaded" << std::endl;
        std::cout << "\nCONTROLS:" << std::endl;
        std::cout << "C: Start collecting (enter object name)" << std::endl;
        std::cout << "S: Stop collecting and save current session" << std::endl;
        std::cout << "F: Finalize and save hybrid database" << std::endl;
        std::cout << "R: Reset new collection data" << std::endl;
        std::cout << "ESC/Q: Quit" << std::endl;
        
        cv::Mat frame, processed, thresholded;
        
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
            
            // Process frame
            cv::Mat preprocessed = thresholder.preprocessImage(frame, true, 3);
            thresholded = thresholder.kmeansThreshold(preprocessed, 16);
            cv::Mat cleaned = morphFilter.cleanupObjects(thresholded, 3, 5);
            
            // Connected components analysis
            cv::Mat labels, stats, centroids;
            int numComponents = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids, 8, CV_32S);
            
            // Process regions
            cv::Mat result = frame.clone();
            std::vector<RegionFeatures> detectedRegions;
            
            for (int i = 1; i < numComponents; i++) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                int left = stats.at<int>(i, cv::CC_STAT_LEFT);
                int top = stats.at<int>(i, cv::CC_STAT_TOP);
                int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
                int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
                
                // Filter regions
                bool touchesBoundary = (left <= 2) || (top <= 2) || 
                                     (left + width >= frame.cols - 2) || 
                                     (top + height >= frame.rows - 2);
                
                if (area > 500 && !touchesBoundary) {
                    // Extract features
                    std::string imageName = "realtime_" + std::to_string(newCollectedData.size());
                    RegionFeatures feature = extractRegionFeatures(cleaned, i, labels, stats, centroids, 
                                                                  currentObjectName, imageName);
                    
                    if (feature.area > 0) {
                        detectedRegions.push_back(feature);
                        
                        // Collect data if in collection mode
                        if (collectingData && !currentObjectName.empty()) {
                            newCollectedData.push_back(feature);
                            sampleCount++;
                        }
                        
                        // Visualize
                        visualizeDetection(result, feature, collectingData);
                    }
                }
            }
            
            // Display information
            std::string fpsText = "FPS: " + std::to_string(fps).substr(0, 5);
            cv::putText(result, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            std::string existingText = "Existing DB: " + std::to_string(existingDatabase.size());
            cv::putText(result, existingText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            std::string newText = "New Samples: " + std::to_string(newCollectedData.size());
            cv::putText(result, newText, cv::Point(10, 85), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            std::string regionsText = "Regions: " + std::to_string(detectedRegions.size());
            cv::putText(result, regionsText, cv::Point(10, 110), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            if (collectingData) {
                std::string collectingText = "COLLECTING: " + currentObjectName + " (" + std::to_string(sampleCount) + ")";
                cv::putText(result, collectingText, cv::Point(10, 140), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            }
            
            // Display results
            cv::imshow("Hybrid Data Collection", result);
            cv::imshow("Binary Regions", cleaned);
            
            // Handle key presses
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27 || key == 'q' || key == 'Q') break; // ESC or Q
            
            if (key == 'c' || key == 'C') {
                std::cout << "\nEnter object name for collection: ";
                std::cin.ignore();
                std::getline(std::cin, currentObjectName);
                if (!currentObjectName.empty()) {
                    collectingData = true;
                    sampleCount = 0;
                    std::cout << "Started collecting data for: " << currentObjectName << std::endl;
                }
            }
            
            if (key == 's' || key == 'S') {
                if (collectingData) {
                    collectingData = false;
                    std::cout << "Stopped collecting. Collected " << sampleCount << " samples for " << currentObjectName << std::endl;
                    currentObjectName = "";
                    sampleCount = 0;
                }
            }
            
            if (key == 'f' || key == 'F') {
                // Finalize and save hybrid database
                std::string outputPath = "results/hybrid_training_database.csv";
                saveHybridDatabase(outputPath);
                
                // Also backup with timestamp
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::string timestamp = std::to_string(time_t);
                std::string backupPath = "results/hybrid_database_" + timestamp + ".csv";
                saveHybridDatabase(backupPath);
                
                std::cout << "Hybrid database finalized and saved!" << std::endl;
            }
            
            if (key == 'r' || key == 'R') {
                newCollectedData.clear();
                collectingData = false;
                currentObjectName = "";
                sampleCount = 0;
                std::cout << "Reset new collection data" << std::endl;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();
        
        // Auto-save on exit if we have new data
        if (!newCollectedData.empty()) {
            std::cout << "\nAuto-saving collected data..." << std::endl;
            saveHybridDatabase("results/hybrid_training_database.csv");
        }
        
        std::cout << "Data collection session ended." << std::endl;
    }
};

int main() {
    std::cout << "=== TASK 5: HYBRID TRAINING DATA COLLECTION ===" << std::endl;
    std::cout << "Combining Static Images + Real-time Video Collection" << std::endl;
    
    HybridDataCollector collector;
    collector.run();
    
    return 0;
}