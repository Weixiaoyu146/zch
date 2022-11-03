import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
import os
from Dataset_filter import *

Cases = 708


# 读取图片
def load_pairs(path):
    images = []
    jpg_num = int(len(os.listdir(path)) / 2)  # 病例个数
    print("病例个数：" + str(jpg_num))

    # 遍历图片`
    for i in range(1, jpg_num + 1):
        pathA = os.path.join(path, str(i) + 'a.jpg')
        pathP = os.path.join(path, str(i) + 'p.jpg')
        imageA = cv.imread(pathA, cv.IMREAD_GRAYSCALE)  # 读取灰度图
        imageP = cv.imread(pathP, cv.IMREAD_GRAYSCALE)
        images.append([imageA, imageP])

    return images, jpg_num


# 读取图片和索引
def load_images(path):
    images = []
    indexs = get_indexs(path)

    for i in indexs:
        pathA = os.path.join(path, str(i) + 'a.jpg')
        pathP = os.path.join(path, str(i) + 'p.jpg')
        imageA = cv.imread(pathA, cv.IMREAD_GRAYSCALE)  # 读取灰度图
        imageP = cv.imread(pathP, cv.IMREAD_GRAYSCALE)
        images.append([imageA, imageP, i])  # 前身图，后身图，图片id
        # print(pathA + str(i))

    return images


# 图像增强
def image_enhance(image, id):
    # row_histogram(image, 'origin')

    # 正则化
    image_norm = cv.normalize(image, dst=None, alpha=400, beta=15, norm_type=cv.NORM_MINMAX)
    # row_histogram(image_norm, 'norm')
    # cv.imshow('enhance', image_norm)

    # 均值滤波
    image_mean = cv.blur(image_norm, (3, 3), 3)
    # row_histogram(image_mean, 'mean')
    # cv.imshow('mean', image_mean)

    # images = np.hstack((image, image_norm, image_mean))
    # cv.imshow(str(id) + '-image-norm-mean', images)

    # # 锐化
    # k = np.array(([-1, 0, -1], [0, 5, 0], [-1, 0, -1]), np.float32)
    # image_filter = cv.filter2D(image_mean, -1, kernel=k)
    # row_histogram(image_filter, 'enhance')

    # # 反变换
    # image_reverse = 255 - image
    # cv.imshow('reverse', image_reverse)
    #
    # image_r = ~image
    # cv.imshow('r', image_r)
    # print(max(map(max, image_mean)))

    return image_mean


# 提取骨架
def get_bone(image):
    # row_histogram(image, 'origin')

    # 直方图均衡化
    image = cv.equalizeHist(image)
    # row_histogram(image, 'his')

    # 二值化阈值分割
    # _, binary = cv.threshold(image, 84, 255, cv.THRESH_TOZERO)
    _, binary = cv.threshold(image, 100, 255, cv.THRESH_TOZERO)
    # row_histogram(binary, 'binary')

    # 方案二：形态学开闭
    # binary = morphological_opening_and_closing(binary, image)

    # 开运算
    kernel = np.ones((5, 5), np.uint8)
    image_open = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)
    # row_histogram(image_open, 'open')

    # 均值滤波
    image_blur = cv.blur(image_open, (5, 5), 5)
    # row_histogram(image_blur, 'blur')

    # 锐化
    k = np.array(([-1, 0, -1], [0, 5, 0], [-1, 0, -1]), np.float32)
    image_filter = cv.filter2D(image_blur, -1, kernel=k)
    # row_histogram(image_filter, 'filter')

    # 直方图均衡化
    # result = cv.equalizeHist(image_filter)
    # row_histogram(result, 'result')

    return image_filter


# 形态学开闭处理
def morphological_opening_and_closing(binary, image):
    kernel = np.ones((5, 5), np.uint8)

    # 闭运算
    image_close = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=1)
    # cv.imshow('close', image_close)

    # # 高斯滤波
    # image_gaussian = cv.GaussianBlur(image_close, (5,5), 5)
    # cv.imshow('Gaussian', image_gaussian)

    # 均值滤波
    # image_mean = cv.blur(image_close, (5,5), 5)
    # cv.imshow('mean', image_mean)

    # 中值滤波
    image_median = cv.medianBlur(image_close, 5)
    # cv.imshow('median', image_median)

    # 开运算
    image_open = cv.morphologyEx(image_median, cv.MORPH_OPEN, kernel, iterations=1)
    # cv.imshow('open', image_open)

    # 掩码
    image_mask = cv.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=image_open)
    # cv.imshow('mask', image_mask)

    # 高斯滤波
    # image_blur = cv.GaussianBlur(image_mask, (5,5), sigmaX=0.75, sigmaY=0.75)
    # # cv.imshow('Gaussian', image_blur)

    # 直方图均衡化
    # image_blur = cv.equalizeHist(image_mask)
    # cv.imshow('e', image_blur)

    # 二值化阈值分割
    _, image_mor = cv.threshold(image_mask, 40, 255, cv.THRESH_TOZERO)
    # row_histogram(image_mor, 'mor')

    return image_mor


# 直方图绘制
def row_histogram(image, name=''):
    rows = np.sum(image, axis=1) / 256

    hist = np.zeros((image.shape[0], int(max(rows))), dtype=np.uint8)
    count = 0
    # 绘制直方图
    for x in rows:
        for y in range(int(x)):
            hist[count][y] = 255
        count += 1
    # 将骨扫描图像和                                                                                               水平统计直方图放一起便于观察
    image_fuse = np.zeros((image.shape[0], image.shape[1] + hist.shape[1]), dtype=np.uint8)
    image_fuse[:, 0:image.shape[1]] = image
    image_fuse[:, image.shape[1]:] = hist

    # 图片显示
    if name.strip() == '':
        name = 'None'
    cv.imshow(name, image_fuse)

    return


def col_histogram(image, name=''):
    cols = np.sum(image, axis=0) / 256

    hist = np.zeros((int(max(cols)), image.shape[1]), dtype=np.uint8)
    count = 0
    # print(hist.shape)
    # 绘制直方图
    for y in cols:
        for x in range(int(y)):
            hist[x][count] = 255
        count += 1
    # 将骨扫描图像和水平统计直方图放一起便于观察
    image_fuse = np.zeros((image.shape[0] + hist.shape[0], image.shape[1]), dtype=np.uint8)
    image_fuse[0:image.shape[0], :] = image
    image_fuse[image.shape[0]:, :] = hist

    # 图片显示
    if name.strip() == '':
        name = 'None'
    cv.imshow(name, image_fuse)

    return


# def histogram_statistic(image, title):
#     hist = cv.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
#
#     plt.figure()
#     plt.title(title)
#     plt.xlabel('gray level')
#     plt.ylabel('number of pixels')
#     plt.plot(hist)
#     # 　plt.xlim([0, 255])
#     # plt.imshow(image, cmap='gray')
#
#     plt.show()
#
#     return
