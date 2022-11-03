import pydicom
import pymssql
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot
# from ImageProcessing import *

def connect_sqlserver():
    try:
        conn = pymssql.connect(host='127.0.0.1:1433', user='sa', password='123', database='BoneScan', charset="UTF-8")
        cursor = conn.cursor()
    except Exception as e:
        print("连接数据库失败")
        print(e.__class__.__name__, e)
    return conn, cursor

def disconnect_sqlserver(conn, cursor):
    cursor.close()
    conn.close()

def get_dicoms_path(cursor, sql_sentence):
    try:
        cursor.execute(sql_sentence)
    except:
        print("查询失败")
    finally:
        rows = cursor.fetchall()
        return rows

def get_original_images(dicom_image_path, image_dictory, id):
    dicom_image = pydicom.dcmread(dicom_image_path)
    binary_image = dicom_image.pixel_array
    anterior_binary_image = binary_image[0]
    posterior_binary_image = binary_image[1]
    if "ANTERIOR" in dicom_image_path:
        for file in os.listdir(dicom_image_path[:-19]):
            if "POST" in file:
                dicom_image_path_p = dicom_image_path[:-18] + file
                break
        dicom_image_p = pydicom.dcmread(dicom_image_path_p)
        binary_image_p = dicom_image_p.pixel_array
        anterior_binary_image = binary_image
        posterior_binary_image = binary_image_p
    anterior_image = np.uint8(anterior_binary_image)
    posterior_image = np.uint8(posterior_binary_image)

    anterior_image_path = image_dictory + r'/OriginalImages/' + str(id) + 'a.jpg'
    posterior_image_path = image_dictory + r'/OriginalImages/' + str(id) + 'p.jpg'
    # if not os.path.isdir(image_dictory + r'/Image'):
    #     os.makedirs(image_dictory + r'./Image')
    #     os.mkdir(image_dictory + r'/Image' + r'./Original')

    if anterior_image.shape[1] == 512:
        # anterior_image_histogram = np.sum(anterior_image, axis=0)
        # mid_x1 = np.argmax(anterior_image_histogram, axis=0)
        # posterior_image_histogram = np.sum(posterior_image, axis=0)
        # mid_x2 = np.argmax(posterior_image_histogram, axis=0)
        # if abs(mid_x1 - 255.5) <= abs(mid_x2 - 255.5):
        #     mid_x_a = mid_x1
        #     mid_x_p = 511 - mid_x1
        # else:
        #     mid_x_a = 511 - mid_x2
        #     mid_x_p = mid_x2
        # if mid_x_a < 230:
        #     mid_x_a = 256
        #     mid_x_p = 255
        # anterior_image = anterior_image[:, (mid_x_a - 128):(mid_x_a + 128)]
        # posterior_image = posterior_image[:, (mid_x_p - 128):(mid_x_p + 128)]
        anterior_image = anterior_image[:, 128:384]
        posterior_image = posterior_image[:, 127:383]
        # images = np.hstack((anterior_image, posterior_image))
        # cv.imshow(str(id), images)
        # cv.waitKey()
    # cv.imwrite(anterior_image_path, anterior_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    # cv.imwrite(posterior_image_path, posterior_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    return anterior_image, posterior_image

def gammaTransform(gamma, anterior_image, posterior_image, image_dictory, id):
    # 伽马变换
    anterior_gamma_image = np.power(anterior_image/float(np.max(anterior_image)), gamma)
    anterior_gamma_image = np.uint8(anterior_gamma_image * 255)
    posterior_gamma_image = np.power(posterior_image/float(np.max(posterior_image)), gamma)
    posterior_gamma_image = np.uint8(posterior_gamma_image * 255)
    # 二值分割
    _, anterior_binary_image = cv.threshold(anterior_gamma_image, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)
    _, posterior_binary_image = cv.threshold(posterior_gamma_image, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)
    # 开运算
    # kernel = np.ones((5, 5), np.uint8)
    # anterior_open_image = cv.morphologyEx(anterior_binary_image, cv.MORPH_OPEN, kernel, iterations=1)
    # 均值滤波
    # anterior_blur_image = cv.blur(anterior_open_image, (5, 5), 5)
    # 锐化
    # k = np.array(([-1, 0, -1], [0, 5, 0], [-1, 0, -1]), np.float32)
    # anterior_filter_image = cv.filter2D(anterior_blur_image, -1, kernel=k)

    # images = np.hstack((anterior_image, anterior_gamma_image, anterior_binary_image, anterior_open_image, anterior_blur_image, anterior_filter_image))
    # cv.imshow(str(id), images)
    # cv.waitKey()

    anterior_image_path = image_dictory + r'/GammaTransformedImages2/' + str(id) + 'a.jpg'
    posterior_image_path = image_dictory + r'/GammaTransformedImages2/' + str(id) + 'p.jpg'

    # cv.imwrite(anterior_image_path, anterior_binary_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    cv.imwrite(anterior_image_path, anterior_gamma_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    # cv.imwrite(posterior_image_path, posterior_binary_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    cv.imwrite(posterior_image_path, posterior_gamma_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

def ContrastAugmentation(image, id, wc, ww):
    image = np.uint8(image.copy())

    minWindow = float(wc) - 0.5 * float(ww)
    image_norm = (image - minWindow) / float(ww)
    image_norm[image_norm < 0] = 0
    image_norm[image_norm > 1] = 1
    image_norm = (image_norm * 255).astype('uint8')

    return image_norm

def calHistogram(img):
    if(len(img.shape) != 2):
        print("img size error")
        return None
    histogram = {}
    for i1 in range(img.shape[0]):
        for i2 in range(img.shape[1]):
            if histogram.get(img[i1][i2]) is None:
                histogram[img[i1][i2]] = 0
            histogram[img[i1][i2]] += 1
    # normalize
    for key in histogram:
        histogram[key] = float(histogram[key]) * 6 / (img.shape[0]*img.shape[1])
    return histogram

def drawHistoGram(histogram):
    pyplot.figure()
    #设置x轴的最小值，最大值。y轴的最小值，最大值
    pyplot.axis([0, 10, 0, 1])
    #显示网格线
    pyplot.grid(True)
    #key正好就是灰度
    keys = histogram.keys()
    #value是灰度的像素数量，这里是归一化之后的
    values = histogram.values()
    #这里正式绘制直方图
    pyplot.bar(tuple(keys), tuple(values))
    pyplot.show()


if __name__ == '__main__':
    conn, cursor = connect_sqlserver()

    image_dictory = "../../Resources/PreprocessedImages"
    table_name = 'BoneScanImage'
    # sql_sentence = "select Image_path, ID from " + table_name + " where ID >= 2000 and ID < 2600 and Availability = 1"
    sql_sentence = "select Image_path, ID from " + table_name
    dicoms_path_ids = get_dicoms_path(cursor, sql_sentence)
    dicoms_path_ids = list(dicoms_path_ids)

    resolution = set()

    for dicom_path_id in dicoms_path_ids:
        dicom_path, id = dicom_path_id[0], dicom_path_id[1]
        dicom_path = 'BoneScintigraphyImages' + dicom_path[8:]
        try:
            original_anterior_image, original_posterior_image = get_original_images(r'../../Resources/' + dicom_path, image_dictory, id)
        except Exception as e:
            print(10000 + id)
            print(dicom_path)
            print(e.__class__.__name__, e)
            continue

        try:
            gammaTransform(1, original_anterior_image, original_posterior_image, image_dictory, id)
        except Exception as e:
            print(30000 + id)
            print(e.__class__.__name__, e)
            continue


        # try:
        #     preprocessed_anterior_image, preprocessed_posterior_image, preprocessed_anterior_image_mask, preprocessed_posterior_image_mask = cut_preprocessing_image(
        #         original_anterior_image, original_posterior_image, image_dictory, id)
        # except Exception as e:
        #     print(20000 + id)
        #     print(dicom_path)
        #     print(e.__class__.__name__, e)
        #     continue

        resolution.add(original_anterior_image.shape)
        resolution.add(original_posterior_image.shape)
    #
    print(resolution)
    disconnect_sqlserver(conn, cursor)

