cmake_minimum_required(VERSION 3.2.0)

project( Object_Detection )

find_package( OpenCV REQUIRED )
find_package( OpenGL )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(./imagewatch_in_visualStudio)
add_subdirectory(./opencv-haar-boosted)
