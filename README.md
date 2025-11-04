Vision2D - 2D Object Recognition System
A C++/OpenCV pipeline for real-time 2D object recognition using classical computer vision and ONNX-based deep learning embeddings.
Features

Custom thresholding, morphological filtering, connected components analysis, and feature extraction
Real-time camera-based object detection and classification
ONNX ResNet18 embedding classifier for robust recognition
Modular pipeline architecture with individual task executables
Windows build scripts with cross-platform CMake support

Requirements
Build Tools

Windows 10/11 (or cross-platform with C++17 compiler)
CMake ≥ 3.19
Visual Studio 2019/2022 or MSVC Build Tools
C++17 compiler support

Required Dependencies
Before building, download and place these dependencies in the project root:

OpenCV 4.12.0 (Windows build)

Download from opencv.org
Extract to opencv/ folder in project root


ONNX Runtime 1.16+ (Windows x64)

Download from ONNX Runtime releases
Extract to onnxruntime_nupkg/ folder


ResNet18 ONNX Model

Download resnet18-v2-7.onnx from ONNX Model Zoo
Place in project root



Quick Start
Automated Build (Windows)
powershell# From the repository root
.\build_project.bat

# Executables will be placed in bin\Release\
Manual CMake Build
powershellmkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
Applications
After building, the following executables will be available in bin\Release\:

realtime_detection.exe – Real-time camera detection with classification
camera_selector.exe – Select and test available cameras
task3.exe through task9.exe – Individual pipeline stage demonstrations

Controls: Press ESC to exit any application. See on-screen help for task-specific controls.
Project Structure
Vision2D/
├── src/                          # All source code
│   ├── custom_threshold.cpp/.h   # Thresholding algorithms
│   ├── morphological_filter.cpp/.h
│   ├── connected_components.cpp/.h
│   ├── embedding_classifier.cpp/.h
│   ├── enhanced_classifier.cpp/.h
│   └── task*.cpp                 # Individual task executables
├── docs/images/                  # Sample result images
├── CMakeLists.txt                # Build configuration
├── build_project.bat             # Windows build script
├── .gitignore                    # Excludes large dependencies
└── README.md                     # This file

# Dependencies (not tracked in repo):
├── opencv/                       # OpenCV 4.12 runtime
├── onnxruntime_nupkg/            # ONNX Runtime
└── resnet18-v2-7.onnx            # Pre-trained model
Generated folders such as build/, bin/, and results/ are excluded via .gitignore to keep the repository lean.
Sample Results
Show Image
Compass detected with 99% confidence using enhanced embedding classifier
Show Image
Stapler detected with 86% confidence
Tips

Use git lfs if you plan to keep the ONNX model and third-party binaries under version control
For debugging builds, use --config Debug instead of --config Release
Check that all dependencies are in the correct folders before building

License
Third-party components (OpenCV, ONNX Runtime) retain their respective licenses. Project source code is provided as-is for educational purposes.
