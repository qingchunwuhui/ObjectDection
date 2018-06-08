cmake_minimum_required(VERSION 3.2.0)

PROJECT(ObjectDetection)

# 设置添加当前路径
# Set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(MSVC)
	option(OpenCV_Static "Use static OpenCV libraries" OFF)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(OpenCV REQUIRED)

include_directories(. ${OpenCV_INCLUDE_DIRS})

ADD_SUBDIRECTORY(./opencv-haar-boosted)
ADD_SUBDIRECTORY(./imagewatch_in_visualStudio)
ADD_SUBDIRECTORY(./opencv-HOG-SVM)



