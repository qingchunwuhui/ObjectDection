# 数据准备

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import time

startTime = time.clock()

img = cv.imread(
    "E:/VS_Programming/OpenCV Program/Hog-feature/Hog-feature/data/person.png",
    cv.IMREAD_GRAYSCALE)
img1 = np.sqrt(img / float(np.max(img)))*255
# cv.imshow("Image",img)
# cv.imshow("Image1",img1)
cv.waitKey(0)

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
cell_gradient_vector = np.zeros((int(height / cell_size),
                                 int(width / cell_size), bin_size))
print(cell_gradient_vector.shape)


def cell_gradient(cell_magnitude, cell_angle):
    """
    对cell内每个像素用梯度方向在直方图中进行加权投影（映射到固定的角度范围），就可以得到这个cell的梯度方向直方图了，
    就是该cell对应的8维特征向量而梯度大小作为投影的权值

    """
    orientation_centers = [0] * bin_size
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            gradient_strength = cell_magnitude[k][l]
            gradient_angle = cell_angle[k][l]
            min_angle = int(gradient_angle / angle_unit) % bin_size
            max_angle = (min_angle + 1) % bin_size
            mod = gradient_angle % angle_unit
            orientation_centers[min_angle] += (
                gradient_strength * (1 - (mod / angle_unit)))
            orientation_centers[max_angle] += (
                gradient_strength * (mod / angle_unit))
    return orientation_centers


for i in range(cell_gradient_vector.shape[0]):
    for j in range(cell_gradient_vector.shape[1]):
        cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) *
                                            cell_size, j * cell_size:(j + 1) *
                                            cell_size]  #获取一个细胞单元中的所有梯度值

        cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size, j *
                                    cell_size:(j + 1) *
                                    cell_size]  #获取一个细胞单元中的所有梯度方向

        # print(cell_angle.max())

        cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)

# 可视化Cell梯度直方图
hog_image = np.zeros([height, width])
cell_gradient = cell_gradient_vector
cell_width = cell_size / 2
max_mag = np.array(cell_gradient).max()
for x in range(cell_gradient.shape[0]):
    for y in range(cell_gradient.shape[1]):
        cell_grad = cell_gradient[x][y]
        cell_grad /= max_mag  #由于局部光照的变化以及前景-背景对比度的变化，使得梯度强度的变化范围非常大,就需要进行块内归一化梯度直方图
        angle=0
        angle_gap=angle_unit
        for magnitude in cell_grad:
            angle_radian = math.radians(angle)
            x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
            y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
            x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
            y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
            cv.line(hog_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
            angle += angle_gap


# 统计Block的梯度信息
# hog_vector = []
# for i in range(cell_gradient_vector.shape[0] - 1):
#     for j in range(cell_gradient_vector.shape[1] - 1):
#         block_vector = []       #list
#         block_vector.extend(cell_gradient_vector[i][j])
#         block_vector.extend(cell_gradient_vector[i][j + 1])
#         block_vector.extend(cell_gradient_vector[i + 1][j])
#         block_vector.extend(cell_gradient_vector[i + 1][j + 1])
#         mag = lambda vector:math.sqrt(sum(i**2 for i in vector))
#         magnitude = mag(block_vector)
#         if magnitude != 0:
#             normalize = lambda block_vector,magnitude:[element / magnitude for element in block_vector]
#             block_vector = normalize(block_vector, magnitude)       #梯度归一化
#         hog_vector.append(block_vector)

# print(np.array(hog_vector).shape)



endTime= time.clock()
print(endTime-startTime)

plt.imshow(hog_image, cmap=plt.cm.gray)
plt.show()


