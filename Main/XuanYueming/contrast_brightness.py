import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom

# wincenter = 60
# winwidth = 100
# rows = 1024
# cols = 256
# dicom_image_path = []
# file_dir1 = r'../Image2'
# for file_dir2 in os.listdir(file_dir1):
#     image_path = ''
#     for file in os.listdir(file_dir1 + '/' + file_dir2):
#         file_name = file.split(sep='.')
#
#         if file_name[0] == 'ANT001_DS':
#             image_path = file_dir2 + '/' + file
#         elif file_name[0] == 'ANT002_DS':
#             image_path = file_dir2 + '/' + file
#     dicom_image_path.append(file_dir1 + '/' + image_path)
#
# for file_dir in dicom_image_path:
#     dcm = pydicom.dcmread(file_dir)
#     dimg = dcm.pixel_array
#     image_a = dimg[0]
#     # image_p = dimg[1]
#     imageA = np.uint8(image_a)
#     # imageP = np.uint8(image_p)
#     # intercept来自于DICOM文件中的[0028, 1052]
#     # slope来自于DICOM文件中的[0028, 1053]
#
#     img_copy = image_a.copy()
#
#     minWindow = float(wincenter) - 0.5 * float(winwidth)
#     newimg = (img_copy - minWindow) / float(winwidth)
#     newimg[newimg < 0] = 0
#     newimg[newimg > 1] = 1
#     newimg = (newimg * 255).astype('uint8')
#
#     imgs = np.hstack((imageA, newimg))
#     cv.imshow('all', imgs)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

def ContrastAugmentation(image, id, wc, ww):
    image = np.uint8(image.copy())

    minWindow = float(wc) - 0.5 * float(ww)
    image_norm = (image - minWindow) / float(ww)
    image_norm[image_norm < 0] = 0
    image_norm[image_norm > 1] = 1
    image_norm = (image_norm * 255).astype('uint8')

    image_mean = cv.blur(image_norm, (3, 3), 3)

    # images = np.hstack((image, image_norm, image_mean))
    # cv.imshow(str(id), images)
    # print(max(map(max, image_mean)))

    return image_mean





# 绘制直方图函数
def grayHist(img):
    print(img.shape)
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()

def merge(im1, im2):
	a1 = np.array(im1)
	a2 = np.array(im2)
	arr = np.hstack((a1, a2))
	return arr


# img = cv.imread(r"../Image2/1.2.840.40823.1.1.1.6.1539653760.807.314.264/01-左耻骨不录入-右耻骨不录入-左膝关节热点-右膝关节热点.jpg")
# img1 = cv.imdecode(np.fromfile(r"./1ANT001_DS_Frame1.jpg" ,dtype=np.uint8), cv.IMREAD_UNCHANGED)
# img2 = cv.imdecode(np.fromfile(r"./2ANT001_DS_Frame1_1.jpg" ,dtype=np.uint8), cv.IMREAD_UNCHANGED)
# out = 2.0 * img
# # 进行数据截断，大于255的值截断为255
# out[out > 255] = 255
# # 数据类型转换
# out = np.around(out)
# out = out.astype(np.uint8)
# # 分别绘制处理前后的直方图
# grayHist(img)
# grayHist(out)
# cv.imshow("img", img)
# cv.imshow("out", out)
# cv.waitKey()
# Imax=np.max(img)
# Imin=np.min(img)
# print(Imax)
# print(Imin)
# #要输出的最小灰度级和最大灰度级
# Omax,Omin=123,0
# #计算a和b的值 ,测试出*4 能看到人脸
# a=float(Omax-Omin)/(Imax-Imin)
# b=Omin-a*Imin
# #矩阵的线性变换
# out=a*img+b
# #数据类型转换
# out=out.astype(np.uint8)
#显示原图和直方图正规化的效果
# fI=img/255.0
# #伽马变换
# gamma = 2.3
# out = np.power(fI, gamma)
# cv.imshow("I",img)
# cv.imshow("O",out)
# grayHist(img1)
# grayHist(img2)
# cv.waitKey(0)
# cv.destroyAllWindows()


# data_path_all = []
# data_path_abnormal = []
# data_path_normal = []
# file_dir1 = r'../Image2'
# gamma = 1.55
#
# for file_dir2 in os.listdir(file_dir1):
#     data_path = file_dir1 + '/' + file_dir2
#     data_path_all.append(data_path)
#     for file in os.listdir(data_path):
#         file_name = file.split(sep='.')
#         if file_name[-1] == 'xml':
#             data_path_abnormal.append(data_path)
#             break
#
# data_path_normal = [x for x in data_path_all if x not in data_path_abnormal]
#
# for file_dir in data_path_all:
#     for file in os.listdir(file_dir):
#         file_name = file.split(sep='.')
#         if file_name[-1] == 'jpg':
#             img = cv.imdecode(np.fromfile(file_dir + '/' + file, dtype=np.uint8), cv.IMREAD_UNCHANGED)
#             img_nor = img/255.0
#             out = np.power(img_nor, gamma)
#             print(file_dir)
#             print(file_name[0])
#             imgs = np.hstack((img_nor, out))
#             cv.imshow("out", imgs)
#             cv.waitKey(0)
#             cv.destroyAllWindows()
#             new_file_name = ''
#             for file_name_split in file_name[:-1]:
#                 new_file_name += file_name_split
#             # cv.imencode('.jpg', out)[1].tofile(file_dir + '/' + new_file_name + 'abnormal.jpg')
