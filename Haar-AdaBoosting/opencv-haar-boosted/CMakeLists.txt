FILE(GLOB SRC_FILE "./*.cpp")
FILE(GLOB HEAD_FILE "./*.h")

add_executable(opencv_haar_boosted
${SRC_FILE}
${HEAD_FILE})

TARGET_LINK_LIBRARIES(opencv_haar_boosted ${OpenCV_LIBS})



