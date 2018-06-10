add_subdirectory(./opencv_HOG_SVM)
add_subdirectory(./train_HOG)

set_target_properties(opencv_HOG_SVM train_HOG PROPERTIES FOLDER HOG-SVM)