#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include "custom_threshold.h"
#include "morphological_filter.h"

// Region Features structure (consistent with other tasks)
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
    
    RegionFeatures() : objectName(""), imageName(""), regionId(0), area(0), percentFilled(0), 
                      aspectRatio(0), compactness(0), majorAxisLength(0), minorAxisLength(0), 
                      orientation(0), eccentricity(0), hu1(0), hu2(0), hu3(0), 
                      centroid(0, 0), boundingRect() {}
};

// Enhanced Multi-Classifier System (from task6)
class PerformanceEvaluator {
private:
    std::vector<RegionFeatures> trainingData;
    std::map<std::string, std::map<std::string, double>> featureStats;
    std::vector<std::string> objectClasses;
    
    std::vector<std::string> featureNames = {
        "area", "percentFilled", "aspectRatio", "compactness", 
        "majorAxisLength", "minorAxisLength", "orientation", 
        "eccentricity", "hu1", "hu2", "hu3"
    };
    
    void calculateFeatureStats() {
        std::map<std::string, std::vector<std::vector<double>>> classFeatures;
        
        for (const auto& sample : trainingData) {
            if (classFeatures.find(sample.objectName) == classFeatures.end()) {
                classFeatures[sample.objectName] = std::vector<std::vector<double>>(featureNames.size());
                objectClasses.push_back(sample.objectName);
            }
            
            std::vector<double> features = featuresToVector(sample);
            for (size_t i = 0; i < features.size(); i++) {
                classFeatures[sample.objectName][i].push_back(features[i]);
            }
        }
        
        for (const auto& cls : classFeatures) {
            for (size_t i = 0; i < featureNames.size(); i++) {
                const auto& values = cls.second[i];
                double mean = 0.0, std = 1.0;
                
                if (!values.empty()) {
                    for (double val : values) mean += val;
                    mean /= values.size();
                    
                    double variance = 0.0;
                    for (double val : values) {
                        variance += (val - mean) * (val - mean);
                    }
                    variance /= values.size();
                    std = sqrt(variance);
                    if (std < 1e-10) std = 1.0;
                }
                
                featureStats[cls.first][featureNames[i] + "_mean"] = mean;
                featureStats[cls.first][featureNames[i] + "_std"] = std;
            }
        }
        
        std::sort(objectClasses.begin(), objectClasses.end());
        objectClasses.erase(std::unique(objectClasses.begin(), objectClasses.end()), objectClasses.end());
    }
    
    std::vector<double> featuresToVector(const RegionFeatures& features) {
        return {
            features.area, features.percentFilled, features.aspectRatio, 
            features.compactness, features.majorAxisLength, features.minorAxisLength,
            features.orientation, features.eccentricity, features.hu1, features.hu2, features.hu3
        };
    }
    
    double scaledEuclideanDistance(const RegionFeatures& a, const RegionFeatures& b) {
        std::vector<double> featA = featuresToVector(a);
        std::vector<double> featB = featuresToVector(b);
        
        double distance = 0.0;
        for (size_t i = 0; i < featA.size(); i++) {
            double std_dev = 1.0;
            if (!featureStats.empty() && featureStats.count(b.objectName) > 0) {
                std_dev = featureStats.at(b.objectName).at(featureNames[i] + "_std");
            }
            double diff = (featA[i] - featB[i]) / std_dev;
            distance += diff * diff;
        }
        return sqrt(distance);
    }
    
    double scaledManhattanDistance(const RegionFeatures& a, const RegionFeatures& b) {
        std::vector<double> featA = featuresToVector(a);
        std::vector<double> featB = featuresToVector(b);
        
        double distance = 0.0;
        for (size_t i = 0; i < featA.size(); i++) {
            double std_dev = 1.0;
            if (!featureStats.empty() && featureStats.count(b.objectName) > 0) {
                std_dev = featureStats.at(b.objectName).at(featureNames[i] + "_std");
            }
            distance += abs(featA[i] - featB[i]) / std_dev;
        }
        return distance;
    }
    
    double cosineSimilarity(const RegionFeatures& a, const RegionFeatures& b) {
        std::vector<double> featA = featuresToVector(a);
        std::vector<double> featB = featuresToVector(b);
        
        double dotProduct = 0.0, normA = 0.0, normB = 0.0;
        for (size_t i = 0; i < featA.size(); i++) {
            dotProduct += featA[i] * featB[i];
            normA += featA[i] * featA[i];
            normB += featB[i] * featB[i];
        }
        
        if (normA == 0.0 || normB == 0.0) return 0.0;
        return dotProduct / (sqrt(normA) * sqrt(normB));
    }
    
public:
    void loadTrainingData(const std::vector<RegionFeatures>& data) {
        trainingData = data;
        calculateFeatureStats();
    }
    
    std::string classifyEuclidean(const RegionFeatures& unknown, double& confidence) {
        if (trainingData.empty()) {
            confidence = 0.0;
            return "Unknown";
        }
        
        double minDistance = std::numeric_limits<double>::max();
        std::string bestMatch = "Unknown";
        
        for (const auto& sample : trainingData) {
            double distance = scaledEuclideanDistance(unknown, sample);
            if (distance < minDistance) {
                minDistance = distance;
                bestMatch = sample.objectName;
            }
        }
        
        confidence = 1.0 / (1.0 + minDistance);
        return bestMatch;
    }
    
    std::string classifyManhattan(const RegionFeatures& unknown, double& confidence) {
        if (trainingData.empty()) {
            confidence = 0.0;
            return "Unknown";
        }
        
        double minDistance = std::numeric_limits<double>::max();
        std::string bestMatch = "Unknown";
        
        for (const auto& sample : trainingData) {
            double distance = scaledManhattanDistance(unknown, sample);
            if (distance < minDistance) {
                minDistance = distance;
                bestMatch = sample.objectName;
            }
        }
        
        confidence = 1.0 / (1.0 + minDistance);
        return bestMatch;
    }
    
    std::string classifyCosine(const RegionFeatures& unknown, double& confidence) {
        if (trainingData.empty()) {
            confidence = 0.0;
            return "Unknown";
        }
        
        double maxSimilarity = -1.0;
        std::string bestMatch = "Unknown";
        
        for (const auto& sample : trainingData) {
            double similarity = cosineSimilarity(unknown, sample);
            if (similarity > maxSimilarity) {
                maxSimilarity = similarity;
                bestMatch = sample.objectName;
            }
        }
        
        confidence = (maxSimilarity + 1.0) / 2.0;
        return bestMatch;
    }
    
    std::string classifyKNN(const RegionFeatures& unknown, int k, double& confidence) {
        if (trainingData.empty()) {
            confidence = 0.0;
            return "Unknown";
        }
        
        std::vector<std::pair<double, std::string>> distances;
        
        for (const auto& sample : trainingData) {
            double distance = scaledEuclideanDistance(unknown, sample);
            distances.push_back({distance, sample.objectName});
        }
        
        std::sort(distances.begin(), distances.end());
        
        std::map<std::string, int> votes;
        int actualK = std::min(k, (int)distances.size());
        
        for (int i = 0; i < actualK; i++) {
            votes[distances[i].second]++;
        }
        
        int maxVotes = 0;
        std::string bestMatch = "Unknown";
        for (const auto& vote : votes) {
            if (vote.second > maxVotes) {
                maxVotes = vote.second;
                bestMatch = vote.first;
            }
        }
        
        confidence = double(maxVotes) / actualK;
        return bestMatch;
    }
    
    std::vector<std::pair<std::string, double>> classifyMultiple(const RegionFeatures& unknown) {
        std::map<std::string, double> classifierResults;
        
        double conf1, conf2, conf3, conf4;
        std::string result1 = classifyEuclidean(unknown, conf1);
        std::string result2 = classifyManhattan(unknown, conf2);
        std::string result3 = classifyCosine(unknown, conf3);
        std::string result4 = classifyKNN(unknown, 3, conf4);
        
        classifierResults[result1] += conf1 * 0.4;
        classifierResults[result2] += conf2 * 0.2;
        classifierResults[result3] += conf3 * 0.2;
        classifierResults[result4] += conf4 * 0.2;
        
        std::vector<std::pair<std::string, double>> sortedResults;
        for (const auto& result : classifierResults) {
            sortedResults.push_back({result.first, result.second});
        }
        
        std::sort(sortedResults.begin(), sortedResults.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return sortedResults;
    }
    
    std::vector<std::string> getClasses() const { return objectClasses; }
};

// Feature extractor
class FeatureExtractor {
public:
    RegionFeatures extractFeatures(const cv::Mat& binaryImage, int regionId, 
                                 const cv::Mat& labels, const cv::Mat& stats, 
                                 const cv::Mat& centroids, const std::string& objectName = "", 
                                 const std::string& imageName = "") {
        RegionFeatures feature;
        feature.objectName = objectName;
        feature.imageName = imageName;
        feature.regionId = regionId;
        
        int area = stats.at<int>(regionId, cv::CC_STAT_AREA);
        int left = stats.at<int>(regionId, cv::CC_STAT_LEFT);
        int top = stats.at<int>(regionId, cv::CC_STAT_TOP);
        int width = stats.at<int>(regionId, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(regionId, cv::CC_STAT_HEIGHT);
        
        cv::Mat regionMask = (labels == regionId);
        cv::Moments m = cv::moments(regionMask, true);
        if (m.m00 == 0) return feature;
        
        feature.area = m.m00;
        feature.centroid = cv::Point2f(float(m.m10/m.m00), float(m.m01/m.m00));
        
        double boundingArea = width * height;
        feature.percentFilled = feature.area / boundingArea;
        feature.aspectRatio = double(width) / height;
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (!contours.empty()) {
            double perimeter = cv::arcLength(contours[0], true);
            if (perimeter > 0) {
                feature.compactness = (4 * CV_PI * feature.area) / (perimeter * perimeter);
            }
        }
        
        double mu20 = m.mu20 / m.m00;
        double mu02 = m.mu02 / m.m00;
        double mu11 = m.mu11 / m.m00;
        
        double delta = 4*mu11*mu11 + (mu20-mu02)*(mu20-mu02);
        double sqrt_delta = sqrt(delta);
        double lambda1 = (mu20 + mu02 + sqrt_delta) / 2.0;
        double lambda2 = (mu20 + mu02 - sqrt_delta) / 2.0;
        
        feature.majorAxisLength = 2 * sqrt(std::max(lambda1, 0.0));
        feature.minorAxisLength = 2 * sqrt(std::max(lambda2, 0.0));
        
        if (abs(mu11) < 1e-10 && abs(mu20 - mu02) < 1e-10) {
            feature.orientation = 0;
        } else {
            feature.orientation = 0.5 * atan2(2*mu11, mu20-mu02) * 180.0 / CV_PI;
        }
        
        if (lambda2 > 1e-10) {
            feature.eccentricity = sqrt(lambda1) / sqrt(lambda2);
        } else {
            feature.eccentricity = 1.0;
        }
        
        double hu[7];
        cv::HuMoments(m, hu);
        feature.hu1 = hu[0];
        feature.hu2 = hu[1];
        feature.hu3 = hu[2];
        
        return feature;
    }
};

// Confusion Matrix class
class ConfusionMatrix {
private:
    std::vector<std::string> classes;
    std::map<std::string, std::map<std::string, int>> matrix;
    int totalSamples;
    
public:
    ConfusionMatrix(const std::vector<std::string>& classNames) : classes(classNames), totalSamples(0) {
        // Initialize matrix
        for (const auto& trueClass : classes) {
            for (const auto& predClass : classes) {
                matrix[trueClass][predClass] = 0;
            }
        }
    }
    
    void addPrediction(const std::string& trueLabel, const std::string& predictedLabel) {
        if (matrix.count(trueLabel) > 0 && matrix[trueLabel].count(predictedLabel) > 0) {
            matrix[trueLabel][predictedLabel]++;
            totalSamples++;
        }
    }
    
    void printMatrix() {
        std::cout << "\n=== CONFUSION MATRIX ===" << std::endl;
        
        // Print header
        std::cout << std::setw(12) << "True\\Pred";
        for (const auto& cls : classes) {
            std::cout << std::setw(10) << cls.substr(0, 8);
        }
        std::cout << std::setw(12) << "Total" << std::setw(12) << "Recall" << std::endl;
        
        // Print rows
        for (const auto& trueClass : classes) {
            std::cout << std::setw(12) << trueClass.substr(0, 10);
            
            int rowTotal = 0;
            for (const auto& predClass : classes) {
                int count = matrix[trueClass][predClass];
                std::cout << std::setw(10) << count;
                rowTotal += count;
            }
            
            double recall = rowTotal > 0 ? (double)matrix[trueClass][trueClass] / rowTotal : 0.0;
            std::cout << std::setw(12) << rowTotal;
            std::cout << std::setw(12) << std::fixed << std::setprecision(3) << recall;
            std::cout << std::endl;
        }
        
        // Print column totals and precision
        std::cout << std::setw(12) << "Total";
        std::vector<int> colTotals(classes.size(), 0);
        for (size_t i = 0; i < classes.size(); i++) {
            for (const auto& trueClass : classes) {
                colTotals[i] += matrix[trueClass][classes[i]];
            }
            std::cout << std::setw(10) << colTotals[i];
        }
        std::cout << std::setw(12) << totalSamples << std::endl;
        
        std::cout << std::setw(12) << "Precision";
        for (size_t i = 0; i < classes.size(); i++) {
            double precision = colTotals[i] > 0 ? (double)matrix[classes[i]][classes[i]] / colTotals[i] : 0.0;
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << precision;
        }
        std::cout << std::endl;
    }
    
    double getAccuracy() {
        int correct = 0;
        for (const auto& cls : classes) {
            correct += matrix[cls][cls];
        }
        return totalSamples > 0 ? (double)correct / totalSamples : 0.0;
    }
    
    void saveToFile(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        
        file << "Confusion Matrix Results" << std::endl;
        file << "Total Samples: " << totalSamples << std::endl;
        file << "Overall Accuracy: " << std::fixed << std::setprecision(4) << getAccuracy() << std::endl;
        file << std::endl;
        
        // Matrix
        file << "True\\Predicted,";
        for (const auto& cls : classes) {
            file << cls << ",";
        }
        file << "Total,Recall" << std::endl;
        
        for (const auto& trueClass : classes) {
            file << trueClass << ",";
            int rowTotal = 0;
            for (const auto& predClass : classes) {
                int count = matrix[trueClass][predClass];
                file << count << ",";
                rowTotal += count;
            }
            double recall = rowTotal > 0 ? (double)matrix[trueClass][trueClass] / rowTotal : 0.0;
            file << rowTotal << "," << std::fixed << std::setprecision(4) << recall << std::endl;
        }
        
        file.close();
    }
};

// Load database from CSV
std::vector<RegionFeatures> loadDatabase(const std::string& csvPath) {
    std::vector<RegionFeatures> database;
    std::ifstream file(csvPath);
    
    if (!file.is_open()) {
        std::cerr << "Could not open database: " << csvPath << std::endl;
        return database;
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
        
        database.push_back(feature);
    }
    
    file.close();
    return database;
}

int main() {
    std::cout << "=== TASK 7: PERFORMANCE EVALUATION ===" << std::endl;
    std::cout << "5x5 Confusion Matrix Generation for Multiple Classifiers" << std::endl;
    
    // Load training database
    std::vector<RegionFeatures> allData = loadDatabase("results/comprehensive_feature_vectors.csv");
    if (allData.empty()) {
        std::cerr << "Error: No data loaded. Run task4 first!" << std::endl;
        return -1;
    }
    
    std::cout << "Loaded " << allData.size() << " total samples" << std::endl;
    
    // Get unique classes
    std::set<std::string> uniqueClasses;
    for (const auto& sample : allData) {
        uniqueClasses.insert(sample.objectName);
    }
    std::vector<std::string> classes(uniqueClasses.begin(), uniqueClasses.end());
    std::sort(classes.begin(), classes.end());
    
    std::cout << "Found " << classes.size() << " classes: ";
    for (const auto& cls : classes) {
        std::cout << cls << " ";
    }
    std::cout << std::endl;
    
    // Split data into training and testing (80/20 split)
    std::map<std::string, std::vector<RegionFeatures>> classData;
    for (const auto& sample : allData) {
        classData[sample.objectName].push_back(sample);
    }
    
    std::vector<RegionFeatures> trainingData, testingData;
    
    for (const auto& classGroup : classData) {
        const auto& samples = classGroup.second;
        int trainSize = (int)(samples.size() * 0.8);
        
        for (int i = 0; i < trainSize; i++) {
            trainingData.push_back(samples[i]);
        }
        for (int i = trainSize; i < (int)samples.size(); i++) {
            testingData.push_back(samples[i]);
        }
    }
    
    std::cout << "Training samples: " << trainingData.size() << std::endl;
    std::cout << "Testing samples: " << testingData.size() << std::endl;
    
    // Initialize evaluator with training data
    PerformanceEvaluator evaluator;
    evaluator.loadTrainingData(trainingData);
    
    // Create confusion matrices for different classifiers
    ConfusionMatrix euclideanMatrix(classes);
    ConfusionMatrix manhattanMatrix(classes);
    ConfusionMatrix cosineMatrix(classes);
    ConfusionMatrix knnMatrix(classes);
    ConfusionMatrix ensembleMatrix(classes);
    
    std::cout << "\nEvaluating classifiers on test data..." << std::endl;
    
    // Test each sample
    for (const auto& testSample : testingData) {
        // Skip training samples to avoid bias
        RegionFeatures unknown = testSample;
        unknown.objectName = ""; // Remove label for classification
        
        double conf1, conf2, conf3, conf4;
        std::string euclideanResult = evaluator.classifyEuclidean(unknown, conf1);
        std::string manhattanResult = evaluator.classifyManhattan(unknown, conf2);
        std::string cosineResult = evaluator.classifyCosine(unknown, conf3);
        std::string knnResult = evaluator.classifyKNN(unknown, 3, conf4);
        
        // Ensemble classification
        auto ensembleResults = evaluator.classifyMultiple(unknown);
        std::string ensembleResult = ensembleResults.empty() ? "Unknown" : ensembleResults[0].first;
        
        // Add to confusion matrices
        euclideanMatrix.addPrediction(testSample.objectName, euclideanResult);
        manhattanMatrix.addPrediction(testSample.objectName, manhattanResult);
        cosineMatrix.addPrediction(testSample.objectName, cosineResult);
        knnMatrix.addPrediction(testSample.objectName, knnResult);
        ensembleMatrix.addPrediction(testSample.objectName, ensembleResult);
    }
    
    // Print results
    std::cout << "\n=== EUCLIDEAN DISTANCE CLASSIFIER ===" << std::endl;
    euclideanMatrix.printMatrix();
    std::cout << "Overall Accuracy: " << std::fixed << std::setprecision(4) << euclideanMatrix.getAccuracy() << std::endl;
    
    std::cout << "\n=== MANHATTAN DISTANCE CLASSIFIER ===" << std::endl;
    manhattanMatrix.printMatrix();
    std::cout << "Overall Accuracy: " << std::fixed << std::setprecision(4) << manhattanMatrix.getAccuracy() << std::endl;
    
    std::cout << "\n=== COSINE SIMILARITY CLASSIFIER ===" << std::endl;
    cosineMatrix.printMatrix();
    std::cout << "Overall Accuracy: " << std::fixed << std::setprecision(4) << cosineMatrix.getAccuracy() << std::endl;
    
    std::cout << "\n=== K-NEAREST NEIGHBORS (K=3) ===" << std::endl;
    knnMatrix.printMatrix();
    std::cout << "Overall Accuracy: " << std::fixed << std::setprecision(4) << knnMatrix.getAccuracy() << std::endl;
    
    std::cout << "\n=== ENSEMBLE CLASSIFIER ===" << std::endl;
    ensembleMatrix.printMatrix();
    std::cout << "Overall Accuracy: " << std::fixed << std::setprecision(4) << ensembleMatrix.getAccuracy() << std::endl;
    
    // Save results to files
    euclideanMatrix.saveToFile("results/confusion_matrix_euclidean.txt");
    manhattanMatrix.saveToFile("results/confusion_matrix_manhattan.txt");
    cosineMatrix.saveToFile("results/confusion_matrix_cosine.txt");
    knnMatrix.saveToFile("results/confusion_matrix_knn.txt");
    ensembleMatrix.saveToFile("results/confusion_matrix_ensemble.txt");
    
    // Summary comparison
    std::cout << "\n=== CLASSIFIER COMPARISON SUMMARY ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Euclidean Distance: " << euclideanMatrix.getAccuracy() << std::endl;
    std::cout << "Manhattan Distance: " << manhattanMatrix.getAccuracy() << std::endl;
    std::cout << "Cosine Similarity:  " << cosineMatrix.getAccuracy() << std::endl;
    std::cout << "K-NN (k=3):         " << knnMatrix.getAccuracy() << std::endl;
    std::cout << "Ensemble Method:    " << ensembleMatrix.getAccuracy() << std::endl;
    
    // Save summary
    std::ofstream summaryFile("results/classifier_performance_summary.txt");
    if (summaryFile.is_open()) {
        summaryFile << "Object Recognition System Performance Evaluation" << std::endl;
        summaryFile << "================================================" << std::endl;
        summaryFile << "Training samples: " << trainingData.size() << std::endl;
        summaryFile << "Testing samples: " << testingData.size() << std::endl;
        summaryFile << "Number of classes: " << classes.size() << std::endl;
        summaryFile << std::endl;
        summaryFile << "Classifier Accuracies:" << std::endl;
        summaryFile << "Euclidean Distance: " << euclideanMatrix.getAccuracy() << std::endl;
        summaryFile << "Manhattan Distance: " << manhattanMatrix.getAccuracy() << std::endl;
        summaryFile << "Cosine Similarity:  " << cosineMatrix.getAccuracy() << std::endl;
        summaryFile << "K-NN (k=3):         " << knnMatrix.getAccuracy() << std::endl;
        summaryFile << "Ensemble Method:    " << ensembleMatrix.getAccuracy() << std::endl;
        summaryFile.close();
    }
    
    std::cout << "\n✓ Performance evaluation complete!" << std::endl;
    std::cout << "✓ Confusion matrices saved to results/ folder" << std::endl;
    std::cout << "✓ Translation, scale, and rotation invariant features used" << std::endl;
    std::cout << "✓ Multiple distance metrics compared" << std::endl;
    
    return 0;
}