#include "embedding_classifier.h"
#include "enhanced_classifier.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <set>

// Simple utilities function implementations (since external files may not be available)
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug) {
    const int ORNet_size = 224; // expected network input size
    cv::Mat blob;
    cv::Mat resized;

    cv::resize(src, resized, cv::Size(ORNet_size, ORNet_size));
	
    cv::dnn::blobFromImage(resized, // input image
                          blob, // output array
                          (1.0/255.0) * (1/0.226), // scale factor
                          cv::Size(ORNet_size, ORNet_size), // resize the image to this
                          cv::Scalar(124, 116, 104),   // subtract mean prior to scaling
                          true, // swapRB
                          false,  // center crop after scaling short side to size
                          CV_32F); // output depth/type

    net.setInput(blob);
    embedding = net.forward("onnx_node!resnetv22_flatten0_reshape0");

    if(debug) {
        std::cout << embedding << std::endl;
    }

    return 0;
}

void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, int cx, int cy, 
                       float theta, float minE1, float maxE1, float minE2, float maxE2, int debug) {
    // Rotate the image to align the primary region with the x-axis
    cv::Mat rotatedImage;
    cv::Mat M;

    M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), -theta*180/CV_PI, 1.0);
    int largest = frame.cols > frame.rows ? frame.cols : frame.rows;
    largest = (int)(1.414 * largest);
    cv::warpAffine(frame, rotatedImage, M, cv::Size(largest, largest));

    if(debug) {
        cv::imshow("rotated", rotatedImage);
    }

    int left = cx + (int)minE1;
    int top = cy - (int)maxE2;
    int width = (int)maxE1 - (int)minE1;
    int height = (int)maxE2 - (int)minE2;

    // Bounds check the ROI
    if(left < 0) {
        width += left;
        left = 0;
    }
    if(top < 0) {
        height += top;
        top = 0;
    }
    if(left + width >= rotatedImage.cols) {
        width = (rotatedImage.cols-1) - left;
    }
    if(top + height >= rotatedImage.rows) {
        height = (rotatedImage.rows-1) - top;
    }

    if(debug) {
        printf("ROI box: %d %d %d %d\n", left, top, width, height);
    }

    // Crop the image to the bounding box of the object
    cv::Rect objroi(left, top, width, height);
    cv::rectangle(rotatedImage, cv::Point2d(objroi.x, objroi.y), 
                 cv::Point2d(objroi.x+objroi.width, objroi.y+objroi.height), 200, 4);

    // Extract the image
    cv::Mat extractedImage(rotatedImage, objroi);

    if(debug) {
        cv::imshow("extracted", extractedImage);
    }
    
    extractedImage.copyTo(embimage);
}

EmbeddingClassifier::EmbeddingClassifier() : modelLoaded(false) {
    std::cout << "=== TASK 9: ONE-SHOT EMBEDDING CLASSIFIER ===" << std::endl;
    std::cout << "CNN-based embedding system using ResNet18" << std::endl;
}

EmbeddingClassifier::~EmbeddingClassifier() {
    // Cleanup if needed
}

bool EmbeddingClassifier::loadResNetModel(const std::string& modelPath) {
    try {
        std::cout << "Loading ResNet18 model from: " << modelPath << std::endl;
        resnetModel = cv::dnn::readNetFromONNX(modelPath);
        
        if (resnetModel.empty()) {
            std::cerr << "Error: Could not load ResNet18 model from " << modelPath << std::endl;
            return false;
        }
        
        // Set backend and target
        resnetModel.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        resnetModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        modelLoaded = true;
        std::cout << "✓ ResNet18 model loaded successfully" << std::endl;
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception loading model: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception loading model: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat EmbeddingClassifier::computeEmbedding(const cv::Mat& roi) {
    if (!modelLoaded) {
        std::cerr << "Error: ResNet model not loaded!" << std::endl;
        return cv::Mat();
    }
    
    if (roi.empty()) {
        std::cerr << "Error: Empty ROI provided for embedding computation" << std::endl;
        return cv::Mat();
    }
    
    cv::Mat embedding;
    cv::Mat processedROI = preprocessROI(roi);
    
    // Use the utilities function to get embedding
    int result = getEmbedding(processedROI, embedding, resnetModel, 0);
    
    if (result != 0) {
        std::cerr << "Error computing embedding" << std::endl;
        return cv::Mat();
    }
    
    return embedding;
}

cv::Mat EmbeddingClassifier::preprocessROI(const cv::Mat& roi, int targetSize) {
    cv::Mat processed;
    
    // Convert to 3-channel if needed
    if (roi.channels() == 1) {
        cv::cvtColor(roi, processed, cv::COLOR_GRAY2BGR);
    } else {
        roi.copyTo(processed);
    }
    
    // Resize to target size (224x224 for ResNet18)
    cv::resize(processed, processed, cv::Size(targetSize, targetSize));
    
    return processed;
}

cv::Mat EmbeddingClassifier::extractRotatedROI(const cv::Mat& frame, const RegionFeatures& features) {
    cv::Mat embeddingImage;
    
    // Extract region parameters
    int cx = static_cast<int>(features.centroid.x);
    int cy = static_cast<int>(features.centroid.y);
    float theta = static_cast<float>(features.orientation * CV_PI / 180.0); // Convert to radians
    
    // Estimate bounding extents from major/minor axis lengths
    float halfMajor = static_cast<float>(features.majorAxisLength / 2.0);
    float halfMinor = static_cast<float>(features.minorAxisLength / 2.0);
    
    // Create extents along primary and secondary axes
    float minE1 = -halfMajor;
    float maxE1 = halfMajor;
    float minE2 = -halfMinor;
    float maxE2 = halfMinor;
    
    // Use utilities function to prepare embedding image
    cv::Mat frameCopy;
    frame.copyTo(frameCopy);
    prepEmbeddingImage(frameCopy, embeddingImage, cx, cy, theta, minE1, maxE1, minE2, maxE2, 0);
    
    return embeddingImage;
}

void EmbeddingClassifier::addTrainingEmbedding(const std::string& objectName, const std::string& imageName, 
                                              const cv::Mat& frame, const RegionFeatures& features) {
    if (!modelLoaded) {
        std::cerr << "Error: Cannot add training embedding - model not loaded" << std::endl;
        return;
    }
    
    // Extract and rotate ROI
    cv::Mat roi = extractRotatedROI(frame, features);
    
    if (roi.empty()) {
        std::cerr << "Error: Could not extract ROI for " << objectName << std::endl;
        return;
    }
    
    // Compute embedding
    cv::Mat embedding = computeEmbedding(roi);
    
    if (embedding.empty()) {
        std::cerr << "Error: Could not compute embedding for " << objectName << std::endl;
        return;
    }
    
    // Store the training data
    EmbeddingData embData(objectName, imageName, embedding, roi);
    trainingEmbeddings.push_back(embData);
    
    std::cout << "✓ Added embedding for " << objectName << " (size: " << embedding.total() << ")" << std::endl;
}

double EmbeddingClassifier::sumSquaredDifference(const cv::Mat& emb1, const cv::Mat& emb2) {
    if (emb1.size() != emb2.size() || emb1.type() != emb2.type()) {
        return std::numeric_limits<double>::max();
    }
    
    cv::Mat diff;
    cv::subtract(emb1, emb2, diff);
    cv::multiply(diff, diff, diff);
    
    cv::Scalar sum = cv::sum(diff);
    return sum[0];
}

double EmbeddingClassifier::euclideanDistance(const cv::Mat& emb1, const cv::Mat& emb2) {
    return std::sqrt(sumSquaredDifference(emb1, emb2));
}

double EmbeddingClassifier::cosineDistance(const cv::Mat& emb1, const cv::Mat& emb2) {
    // Flatten to 1D vectors
    cv::Mat vec1 = emb1.reshape(1, emb1.total());
    cv::Mat vec2 = emb2.reshape(1, emb2.total());
    
    // Convert to float if needed
    vec1.convertTo(vec1, CV_32F);
    vec2.convertTo(vec2, CV_32F);
    
    // Compute dot product
    double dot = vec1.dot(vec2);
    
    // Compute norms
    double norm1 = cv::norm(vec1);
    double norm2 = cv::norm(vec2);
    
    if (norm1 == 0.0 || norm2 == 0.0) {
        return 1.0; // Maximum distance
    }
    
    // Cosine similarity
    double cosineSim = dot / (norm1 * norm2);
    
    // Convert to distance (1 - similarity)
    return 1.0 - cosineSim;
}

std::string EmbeddingClassifier::classifyOneShot(const cv::Mat& frame, const RegionFeatures& features, double& confidence) {
    if (trainingEmbeddings.empty()) {
        confidence = 0.0;
        return "Unknown";
    }
    
    // Extract ROI and compute embedding
    cv::Mat roi = extractRotatedROI(frame, features);
    if (roi.empty()) {
        confidence = 0.0;
        return "Unknown";
    }
    
    cv::Mat queryEmbedding = computeEmbedding(roi);
    if (queryEmbedding.empty()) {
        confidence = 0.0;
        return "Unknown";
    }
    
    // Find nearest neighbor using sum-squared difference
    double minDistance = std::numeric_limits<double>::max();
    std::string bestMatch = "Unknown";
    
    for (const auto& trainData : trainingEmbeddings) {
        double distance = sumSquaredDifference(queryEmbedding, trainData.embedding);
        
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = trainData.objectName;
        }
    }
    
    // Convert distance to confidence (inverse relationship)
    confidence = 1.0 / (1.0 + minDistance / 1000.0); // Normalized by 1000 for reasonable scale
    
    return bestMatch;
}

std::vector<std::pair<std::string, double>> EmbeddingClassifier::classifyKNearestEmbeddings(
    const cv::Mat& frame, const RegionFeatures& features, int k) {
    
    std::vector<std::pair<std::string, double>> results;
    
    if (trainingEmbeddings.empty()) {
        return results;
    }
    
    // Extract ROI and compute embedding
    cv::Mat roi = extractRotatedROI(frame, features);
    if (roi.empty()) {
        return results;
    }
    
    cv::Mat queryEmbedding = computeEmbedding(roi);
    if (queryEmbedding.empty()) {
        return results;
    }
    
    // Compute distances to all training embeddings
    std::vector<std::pair<double, std::string>> distances;
    
    for (const auto& trainData : trainingEmbeddings) {
        double distance = sumSquaredDifference(queryEmbedding, trainData.embedding);
        distances.push_back({distance, trainData.objectName});
    }
    
    // Sort by distance
    std::sort(distances.begin(), distances.end());
    
    // Take top k results
    int actualK = std::min(k, static_cast<int>(distances.size()));
    
    for (int i = 0; i < actualK; i++) {
        double confidence = 1.0 / (1.0 + distances[i].first / 1000.0);
        results.push_back({distances[i].second, confidence});
    }
    
    return results;
}

std::vector<std::string> EmbeddingClassifier::getClasses() const {
    std::set<std::string> uniqueClasses;
    for (const auto& embData : trainingEmbeddings) {
        uniqueClasses.insert(embData.objectName);
    }
    return std::vector<std::string>(uniqueClasses.begin(), uniqueClasses.end());
}

void EmbeddingClassifier::saveTrainingEmbeddings(const std::string& csvPath) {
    std::ofstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << csvPath << " for writing" << std::endl;
        return;
    }
    
    file << "ObjectName,ImageName,EmbeddingSize\n";
    
    for (size_t i = 0; i < trainingEmbeddings.size(); i++) {
        const auto& embData = trainingEmbeddings[i];
        file << embData.objectName << "," 
             << embData.imageName << ","
             << embData.embedding.total() << "\n";
        
        // Save embedding data to separate binary file
        std::string embPath = "results/embeddings/emb_" + std::to_string(i) + ".bin";
        cv::FileStorage fs(embPath, cv::FileStorage::WRITE);
        fs << "embedding" << embData.embedding;
        fs.release();
    }
    
    file.close();
    std::cout << "✓ Saved " << trainingEmbeddings.size() << " embeddings to " << csvPath << std::endl;
}

void EmbeddingClassifier::analyzeEmbeddingSpace() {
    if (trainingEmbeddings.empty()) {
        std::cout << "No training embeddings to analyze" << std::endl;
        return;
    }
    
    std::cout << "\n=== EMBEDDING SPACE ANALYSIS ===" << std::endl;
    std::cout << "Total embeddings: " << trainingEmbeddings.size() << std::endl;
    
    // Analyze by class
    std::map<std::string, int> classCounts;
    for (const auto& embData : trainingEmbeddings) {
        classCounts[embData.objectName]++;
    }
    
    std::cout << "Class distribution:" << std::endl;
    for (const auto& pair : classCounts) {
        std::cout << "  " << pair.first << ": " << pair.second << " samples" << std::endl;
    }
    
    // Compute intra-class and inter-class distances
    std::map<std::string, std::vector<double>> intraClassDistances;
    std::vector<double> interClassDistances;
    
    for (size_t i = 0; i < trainingEmbeddings.size(); i++) {
        for (size_t j = i + 1; j < trainingEmbeddings.size(); j++) {
            double distance = sumSquaredDifference(trainingEmbeddings[i].embedding, 
                                                  trainingEmbeddings[j].embedding);
            
            if (trainingEmbeddings[i].objectName == trainingEmbeddings[j].objectName) {
                intraClassDistances[trainingEmbeddings[i].objectName].push_back(distance);
            } else {
                interClassDistances.push_back(distance);
            }
        }
    }
    
    // Report statistics
    std::cout << "\nDistance Statistics:" << std::endl;
    
    for (const auto& pair : intraClassDistances) {
        if (!pair.second.empty()) {
            double avgIntra = std::accumulate(pair.second.begin(), pair.second.end(), 0.0) / pair.second.size();
            std::cout << "  " << pair.first << " intra-class avg distance: " << avgIntra << std::endl;
        }
    }
    
    if (!interClassDistances.empty()) {
        double avgInter = std::accumulate(interClassDistances.begin(), interClassDistances.end(), 0.0) / interClassDistances.size();
        std::cout << "  Inter-class avg distance: " << avgInter << std::endl;
    }
}

void EmbeddingClassifier::evaluatePerformance(const std::vector<RegionFeatures>& testData, 
                                             const std::vector<cv::Mat>& testFrames) {
    if (testData.size() != testFrames.size()) {
        std::cerr << "Error: Test data and frames size mismatch" << std::endl;
        return;
    }
    
    std::cout << "\n=== EMBEDDING CLASSIFIER PERFORMANCE EVALUATION ===" << std::endl;
    std::cout << "Testing on " << testData.size() << " samples..." << std::endl;
    
    int correct = 0;
    int total = testData.size();
    std::map<std::string, int> classCorrect;
    std::map<std::string, int> classTotal;
    
    for (size_t i = 0; i < testData.size(); i++) {
        double confidence;
        std::string predicted = classifyOneShot(testFrames[i], testData[i], confidence);
        std::string actual = testData[i].objectName;
        
        classTotal[actual]++;
        
        if (predicted == actual) {
            correct++;
            classCorrect[actual]++;
        }
        
        if (i % 10 == 0) {
            std::cout << "  Progress: " << i << "/" << total << std::endl;
        }
    }
    
    // Overall accuracy
    double accuracy = static_cast<double>(correct) / total;
    std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(3) 
              << accuracy * 100.0 << "% (" << correct << "/" << total << ")" << std::endl;
    
    // Per-class accuracy
    std::cout << "\nPer-class Accuracy:" << std::endl;
    for (const auto& pair : classTotal) {
        std::string className = pair.first;
        int totalClass = pair.second;
        int correctClass = classCorrect[className];
        double classAcc = static_cast<double>(correctClass) / totalClass;
        
        std::cout << "  " << className << ": " << std::fixed << std::setprecision(3) 
                  << classAcc * 100.0 << "% (" << correctClass << "/" << totalClass << ")" << std::endl;
    }
}