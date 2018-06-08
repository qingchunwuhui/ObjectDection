# 数据准备

import cv2 as cv
import numpy as np
img = cv.imread(
    "E:/VS_Programming/OpenCV Program/Hog-feature/Hog-feature/data/person.png",
    cv.IMREAD_GRAYSCALE)
img = np.sqrt(img / float(np.max(img)))
# cv.imshow("Image",img)
# cv.imshow("Image1",img1)
# cv.waitKey(0)

#计算各个像素的梯度

height, width = img.shape
#计算像素点x方向的梯度（1阶导数）
gradient_value_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
#计算像素点y放心的梯度（1阶导数）
gradient_value_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
#计算像素点的梯度大小
gradient_magnitude = cv.addWeighted(gradient_value_x, 0.5, gradient_value_y,
                                    0.5, 0)
#计算像素点的梯度方向
gradient_angle = cv.phase(
    gradient_value_x, gradient_value_y, angleInDegrees=True)
print(gradient_magnitude.shape, gradient_angle.shape)

# 为每个细胞单元构建梯度方向直方图
cell_size = 8
bin_size = 8
angle_unit = 360 / bin_size
gradient_magnitude = abs(gradient_magnitude)
cell_gradient_vector=np.zeros((int(height/cell_size),int(width/cell_size),bin_size))
print(cell_gradient_vector.shape)

def cell_gradient(cell_magnitude,cell_angle):
    pass

for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        cell_magnitude = gradient_magnitude[i*cell_size:(i+1)*cell_size,
                            j*cell_size:(j+1)*cell_size]                #获取一个细胞单元中的所有像素值
        

cv.waitKey(0)
