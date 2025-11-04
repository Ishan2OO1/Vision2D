#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include "connected_components.h"
#include "custom_threshold.h"
#include "morphological_filter.h"

int main() {
    std::cout << "=== TASK 3: CONNECTED COMPONENTS ANALYSIS ===" << std::endl;
    std::cout << "Segmenting thresholded images into individual regions" << std::endl;
    
    // Initialize components
    CustomThreshold thresholder;
    MorphologicalFilter morphFilter;
    ConnectedComponentAnalyzer ccAnalyzer(500, true); // Min area 500, filter boundary regions
    
    // Create results directory if it doesn't exist
    std::filesystem::create_directories("results");
    
    // Test images from train directory
    std::vector<std::string> testImages = {
        "../../train/Bottle1.jpeg",
        "../../train/compass01.jpeg",
        "../../train/keychain01.jpeg"
    };
    
    std::cout << "\nProcessing training images..." << std::endl;
    
    for (const std::string& imagePath : testImages) {
        std::cout << "\n--- Processing: " << imagePath << " ---" << std::endl;
        
        // Load image
        cv::Mat originalImage = cv::imread(imagePath);
        if (originalImage.empty()) {
            std::cout << "Could not load image: " << imagePath << std::endl;
            continue;
        }
        
        std::cout << "Image loaded: " << originalImage.cols << "x" << originalImage.rows << std::endl;
        
        // Step 1: Preprocess and threshold
        cv::Mat preprocessed = thresholder.preprocessImage(originalImage, true, 3);
        cv::Mat thresholded = thresholder.kmeansThreshold(preprocessed, 16);
        
        // Step 2: Clean with morphological operations
        cv::Mat cleaned = morphFilter.cleanupObjects(thresholded, 3, 5);
        
        // Step 3: Connected components analysis
        cv::Mat labeledImage;
        std::vector<RegionInfo> regions;
        
        int numRegions = ccAnalyzer.analyzeComponents(cleaned, labeledImage, regions);
        
        if (numRegions > 0) {
            // Step 4: Visualize results
            cv::Mat regionVisualization = ccAnalyzer.visualizeRegions(originalImage, labeledImage, regions);
            cv::Mat coloredRegions = ccAnalyzer.createColoredRegionMap(labeledImage, regions);
            
            // Find largest and most central regions
            RegionInfo largestRegion = ccAnalyzer.findLargestRegion(regions);
            RegionInfo centralRegion = ccAnalyzer.findMostCentralRegion(regions, originalImage.size());
            
            std::cout << "Largest region: Area=" << largestRegion.area << " pixels" << std::endl;
            std::cout << "Most central region: Area=" << centralRegion.area 
                      << ", Centroid=(" << centralRegion.centroid.x << "," << centralRegion.centroid.y << ")" << std::endl;
            
            // Save results
            std::string baseName = imagePath.substr(imagePath.find_last_of("/\\") + 1);
            baseName = baseName.substr(0, baseName.find_last_of("."));
            
            std::string outputOriginal = "results/" + baseName + "_task3_original.jpg";
            std::string outputThresholded = "results/" + baseName + "_task3_thresholded.jpg";
            std::string outputCleaned = "results/" + baseName + "_task3_cleaned.jpg";
            std::string outputRegions = "results/" + baseName + "_task3_regions.jpg";
            std::string outputColored = "results/" + baseName + "_task3_colored.jpg";
            std::string outputPipeline = "results/" + baseName + "_task3_pipeline.jpg";
            
            cv::imwrite(outputOriginal, originalImage);
            cv::imwrite(outputThresholded, thresholded);
            cv::imwrite(outputCleaned, cleaned);
            cv::imwrite(outputRegions, regionVisualization);
            cv::imwrite(outputColored, coloredRegions);
            
            // Create pipeline visualization (4-panel view)
            cv::Mat pipeline;
            cv::Mat topRow, bottomRow;
            
            // Resize images for consistent display
            cv::Mat origResized, threshResized, cleanedResized, regionsResized;
            cv::Size displaySize(320, 240);
            
            cv::resize(originalImage, origResized, displaySize);
            cv::resize(thresholded, threshResized, displaySize);
            cv::cvtColor(threshResized, threshResized, cv::COLOR_GRAY2BGR);
            cv::resize(cleaned, cleanedResized, displaySize);
            cv::cvtColor(cleanedResized, cleanedResized, cv::COLOR_GRAY2BGR);
            cv::resize(regionVisualization, regionsResized, displaySize);
            
            // Add labels
            cv::putText(origResized, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            cv::putText(threshResized, "Thresholded", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            cv::putText(cleanedResized, "Cleaned", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            cv::putText(regionsResized, "Regions", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
            
            // Combine into single image
            cv::hconcat(origResized, threshResized, topRow);
            cv::hconcat(cleanedResized, regionsResized, bottomRow);
            cv::vconcat(topRow, bottomRow, pipeline);
            
            cv::imwrite(outputPipeline, pipeline);
            
            std::cout << "Results saved:" << std::endl;
            std::cout << "  - " << outputOriginal << std::endl;
            std::cout << "  - " << outputRegions << std::endl;
            std::cout << "  - " << outputColored << std::endl;
            std::cout << "  - " << outputPipeline << std::endl;
            
            // Display results (optional - comment out for batch processing)
            cv::imshow("Task 3: Original", originalImage);
            cv::imshow("Task 3: Regions", regionVisualization);
            cv::imshow("Task 3: Colored Regions", coloredRegions);
            cv::imshow("Task 3: Pipeline", pipeline);
            
            std::cout << "Press any key to continue to next image..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
            
        } else {
            std::cout << "No valid regions found in image." << std::endl;
        }
    }
    
    std::cout << "\n=== TASK 3 COMPLETE ===" << std::endl;
    std::cout << "Connected components analysis results saved to results/ directory" << std::endl;
    std::cout << "\nKey achievements:" << std::endl;
    std::cout << "✓ Segmented binary images into individual regions" << std::endl;
    std::cout << "✓ Filtered regions by size and boundary conditions" << std::endl;
    std::cout << "✓ Calculated region statistics (area, centroid, bounding box)" << std::endl;
    std::cout << "✓ Created colored region maps for visualization" << std::endl;
    std::cout << "✓ Identified largest and most central regions" << std::endl;
    std::cout << "✓ Generated comprehensive pipeline visualizations" << std::endl;
    
    return 0;
}