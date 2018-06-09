FILE(GLOB SRC_FILE "./*.cpp")
FILE(GLOB HEAD_FILE "./*.h")

add_executable(opencv_HOG_SVM
${SRC_FILE}
${HEAD_FILE})

TARGET_LINK_LIBRARIES(opencv_HOG_SVM ${OpenCV_LIBS})



