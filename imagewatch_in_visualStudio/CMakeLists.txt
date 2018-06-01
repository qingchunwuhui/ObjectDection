FILE(GLOB SRC_FILE "./*.cpp")
FILE(GLOB HEAD_FILE "./*.h")

add_executable(ImageWatch_visualStudio
${SRC_FILE}
${HEAD_FILE})

TARGET_LINK_LIBRARIES(ImageWatch_visualStudio ${OpenCV_LIBS})



