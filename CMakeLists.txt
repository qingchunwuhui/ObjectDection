cmake_minimum_required(VERSION 3.2.0)

project( Object_Detection )

find_package( OpenCV REQUIRED )
find_package( OpenGL )

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(./Utility)
add_subdirectory(./Haar-AdaBoosting)
add_subdirectory(./HOG-SVM)
