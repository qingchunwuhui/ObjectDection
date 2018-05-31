cmake_minimum_required(VERSION 3.0.0)
Project(objectDetection)

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

target_link_libraries(opencv_haar_boosted ${OpenCV_LIBS})

