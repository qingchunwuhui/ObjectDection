add_subdirectory(./opencv_HOG_SVM)
add_subdirectory(./train_HOG)
add_subdirectory(./opencv_SVM)

set_target_properties(opencv_HOG_SVM train_HOG opencv_SVM PROPERTIES FOLDER HOG-SVM)