cmake_minimum_required(VERSION 3.19)
project(ObjectRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV configuration
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime configuration
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime_nupkg")
include_directories("${ONNXRUNTIME_DIR}/build/native/include")
link_directories("${ONNXRUNTIME_DIR}/runtimes/win-x64/native")

# Source files
set(COMMON_SOURCES
    src/custom_threshold.cpp
    src/morphological_filter.cpp
    src/connected_components.cpp
)

# Executables
add_executable(camera_selector src/camera_selector.cpp ${COMMON_SOURCES})
target_link_libraries(camera_selector ${OpenCV_LIBS})

add_executable(realtime_detection src/realtime_detection.cpp ${COMMON_SOURCES})
target_link_libraries(realtime_detection ${OpenCV_LIBS} onnxruntime)

add_executable(task3 src/task3.cpp ${COMMON_SOURCES})
target_link_libraries(task3 ${OpenCV_LIBS})

add_executable(task4 src/task4.cpp ${COMMON_SOURCES})
target_link_libraries(task4 ${OpenCV_LIBS})

add_executable(task5 src/task5.cpp ${COMMON_SOURCES})
target_link_libraries(task5 ${OpenCV_LIBS})

add_executable(task6 src/task6.cpp src/embedding_classifier.cpp ${COMMON_SOURCES})
target_link_libraries(task6 ${OpenCV_LIBS} onnxruntime)

add_executable(task7 src/task7.cpp src/embedding_classifier.cpp ${COMMON_SOURCES})
target_link_libraries(task7 ${OpenCV_LIBS} onnxruntime)

add_executable(task9 src/task9.cpp src/enhanced_classifier.cpp src/embedding_classifier.cpp ${COMMON_SOURCES})
target_link_libraries(task9 ${OpenCV_LIBS} onnxruntime)

# Copy DLLs to output directory
if(WIN32)
    file(GLOB OPENCV_DLLS "${CMAKE_SOURCE_DIR}/opencv/build/x64/vc16/bin/*.dll")
    file(GLOB ONNX_DLLS "${ONNXRUNTIME_DIR}/runtimes/win-x64/native/*.dll")
    
    foreach(dll ${OPENCV_DLLS} ${ONNX_DLLS})
        file(COPY ${dll} DESTINATION ${CMAKE_BINARY_DIR}/Release)
        file(COPY ${dll} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
    endforeach()
endif()
