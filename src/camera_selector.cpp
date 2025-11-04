#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class CameraSelector {
private:
    std::vector<int> availableCameras;
    
    void scanForCameras() {
        std::cout << "Scanning for available cameras..." << std::endl;
        availableCameras.clear();
        
        for (int i = 0; i < 10; i++) {
            cv::VideoCapture cap(i);
            if (cap.isOpened()) {
                cv::Mat testFrame;
                cap >> testFrame;
                if (!testFrame.empty()) {
                    // Get camera info
                    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
                    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
                    double fps = cap.get(cv::CAP_PROP_FPS);
                    
                    std::cout << "Camera " << i << " found: " << width << "x" << height 
                              << " @ " << fps << " FPS" << std::endl;
                    availableCameras.push_back(i);
                }
                cap.release();
            }
        }
        
        if (availableCameras.empty()) {
            std::cout << "No working cameras found!" << std::endl;
        } else {
            std::cout << "Found " << availableCameras.size() << " working camera(s)" << std::endl;
        }
    }
    
    void testCamera(int cameraIndex) {
        std::cout << "\nTesting camera " << cameraIndex << "..." << std::endl;
        
        cv::VideoCapture cap(cameraIndex);
        if (!cap.isOpened()) {
            std::cout << "Failed to open camera " << cameraIndex << std::endl;
            return;
        }
        
        // Set camera properties
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        // Get actual properties
        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        
        std::cout << "Camera " << cameraIndex << " configured: " << width << "x" << height 
                  << " @ " << fps << " FPS" << std::endl;
        
        std::string windowName = "Camera " + std::to_string(cameraIndex) + " Test";
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        
        std::cout << "Press ESC to close this camera test, or 'n' to test next camera" << std::endl;
        
        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                std::cout << "Failed to capture frame from camera " << cameraIndex << std::endl;
                break;
            }
            
            // Add camera info overlay
            std::string info = "Camera " + std::to_string(cameraIndex) + " - " + 
                              std::to_string(width) + "x" + std::to_string(height);
            cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, 
                       cv::Scalar(0, 255, 0), 2);
            
            std::string controls = "ESC: Close | N: Next Camera | Q: Quit All";
            cv::putText(frame, controls, cv::Point(10, height - 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                       cv::Scalar(255, 255, 255), 2);
            
            cv::imshow(windowName, frame);
            
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27) { // ESC
                break;
            } else if (key == 'n' || key == 'N') {
                break;
            } else if (key == 'q' || key == 'Q') {
                cap.release();
                cv::destroyWindow(windowName);
                return;
            }
        }
        
        cap.release();
        cv::destroyWindow(windowName);
    }
    
public:
    void run() {
        std::cout << "=== CAMERA SELECTION UTILITY ===" << std::endl;
        std::cout << "This tool helps you identify available cameras and their capabilities." << std::endl;
        std::cout << "Use this before running the main detection system." << std::endl << std::endl;
        
        scanForCameras();
        
        if (availableCameras.empty()) {
            std::cout << "\nNo cameras available. Please check:" << std::endl;
            std::cout << "1. Camera is connected properly" << std::endl;
            std::cout << "2. Camera drivers are installed" << std::endl;
            std::cout << "3. Camera is not being used by another application" << std::endl;
            std::cout << "4. Windows camera permissions are enabled" << std::endl;
            return;
        }
        
        std::cout << "\n=== CAMERA TEST OPTIONS ===" << std::endl;
        std::cout << "1. Test all cameras sequentially" << std::endl;
        std::cout << "2. Test specific camera" << std::endl;
        std::cout << "3. Show camera list only" << std::endl;
        std::cout << "4. Quit" << std::endl;
        
        int choice;
        std::cout << "\nSelect option (1-4): ";
        std::cin >> choice;
        
        switch (choice) {
            case 1:
                // Test all cameras
                std::cout << "\nTesting all cameras. Press 'n' to move to next camera or ESC to skip." << std::endl;
                for (int cameraIndex : availableCameras) {
                    testCamera(cameraIndex);
                }
                break;
                
            case 2: {
                // Test specific camera
                std::cout << "\nAvailable cameras: ";
                for (size_t i = 0; i < availableCameras.size(); i++) {
                    std::cout << availableCameras[i];
                    if (i < availableCameras.size() - 1) std::cout << ", ";
                }
                std::cout << std::endl;
                
                int selectedCamera;
                std::cout << "Enter camera number to test: ";
                std::cin >> selectedCamera;
                
                if (std::find(availableCameras.begin(), availableCameras.end(), selectedCamera) != availableCameras.end()) {
                    testCamera(selectedCamera);
                } else {
                    std::cout << "Invalid camera number!" << std::endl;
                }
                break;
            }
            
            case 3:
                // Show list only
                std::cout << "\nAvailable cameras summary:" << std::endl;
                for (int cameraIndex : availableCameras) {
                    cv::VideoCapture cap(cameraIndex);
                    if (cap.isOpened()) {
                        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
                        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
                        int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
                        int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
                        double fps = cap.get(cv::CAP_PROP_FPS);
                        std::cout << "  Camera " << cameraIndex << ": " << width << "x" << height 
                                  << " @ " << fps << " FPS" << std::endl;
                        cap.release();
                    }
                }
                break;
                
            case 4:
                std::cout << "Exiting camera selector." << std::endl;
                break;
                
            default:
                std::cout << "Invalid option!" << std::endl;
                break;
        }
        
        std::cout << "\n=== RECOMMENDATIONS ===" << std::endl;
        std::cout << "For best object detection results:" << std::endl;
        std::cout << "1. Use the highest resolution camera available" << std::endl;
        std::cout << "2. Ensure good lighting conditions" << std::endl;
        std::cout << "3. Use a clean white background" << std::endl;
        std::cout << "4. Position camera to minimize shadows" << std::endl;
        std::cout << "5. Avoid camera shake or vibration" << std::endl;
    }
};

int main() {
    CameraSelector selector;
    selector.run();
    return 0;
}