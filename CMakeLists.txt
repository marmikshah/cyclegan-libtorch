cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(artium)

# Libtorch and Torchvision directories
list(APPEND CMAKE_PREFIX_PATH "/opt/libs/libtorch")
list(APPEND CMAKE_PREFIX_PATH "/opt/libs/vision/")
find_package(TorchVision REQUIRED)


# OpenCV Headers. (Installed from source libopencv-dev)
list(APPEND CMAKE_PREFIX_PATH "/opt/libs/opencv/build/")
set(OpenCV_DIR /usr/local/include/opencv4/opencv2/)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Add Program files

add_executable(artium ./src/main.cpp)
set_property(TARGET artium PROPERTY CXX_STANDARD 20)
target_include_directories(artium PRIVATE ./include/)
target_link_libraries(artium PUBLIC ${OpenCV_LIBS} )
target_link_libraries(artium PUBLIC TorchVision::TorchVision)