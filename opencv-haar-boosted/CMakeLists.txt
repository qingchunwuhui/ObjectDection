cmake_minimum_required(VERSION 3.2.0)

PROJECT(ObjectDetection)

# 设置添加当前路径
# Set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(MSVC)
	option(OpenCV_Static "Use static OpenCV libraries" OFF)
endif()

find_package(OpenCV REQUIRED)

include_directories(. ${OpenCV_INCLUDE_DIRS})

FILE(GLOB SRC_FILE "./*.cpp")
FILE(GLOB HEAD_FILE "./*.h")

add_executable(opencv_haar_boosted
${SRC_FILE}
${HEAD_FILE})

set_target_properties(opencv_haar_boosted PROPERTIES LINKER_LANGUAGE CXX)

TARGET_LINK_LIBRARIES(opencv_haar_boosted ${OpenCV_LIBS})



