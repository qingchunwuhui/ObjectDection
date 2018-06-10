FILE(GLOB SRC_FILE "./*.cpp")
FILE(GLOB HEAD_FILE "./*.h")

add_executable(train_HOG
${SRC_FILE}
${HEAD_FILE})

TARGET_LINK_LIBRARIES(train_HOG ${OpenCV_LIBS})



