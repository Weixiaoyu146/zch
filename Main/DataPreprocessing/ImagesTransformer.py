import os

import cv2 as cv
import numpy as np

IMAGE_DIR = "../../Resources/Dataset/Images-Gray"


def gammaTransform(gamma, image, image_dictory, id):
    # 伽马变换
    gamma_image = np.power(image / float(np.max(image)), gamma)
    gamma_image = np.uint8(gamma_image * 255)
    # 二值分割
    _, binary_image = cv.threshold(gamma_image, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)

    image_path = os.path.join(image_dictory, id)

    # cv.imwrite(image_path, binary_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    cv.imwrite(image_path, gamma_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    return gamma_image


# def gammaTransform(gamma, anterior_image, posterior_image, image_dictory, id):
#     # 伽马变换
#     anterior_gamma_image = np.power(anterior_image / float(np.max(anterior_image)), gamma)
#     anterior_gamma_image = np.uint8(anterior_gamma_image * 255)
#     posterior_gamma_image = np.power(posterior_image / float(np.max(posterior_image)), gamma)
#     posterior_gamma_image = np.uint8(posterior_gamma_image * 255)
#     # 二值分割
#     _, anterior_binary_image = cv.threshold(anterior_gamma_image, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)
#     _, posterior_binary_image = cv.threshold(posterior_gamma_image, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)
#     # 开运算
#     # kernel = np.ones((5, 5), np.uint8)
#     # anterior_open_image = cv.morphologyEx(anterior_binary_image, cv.MORPH_OPEN, kernel, iterations=1)
#     # 均值滤波
#     # anterior_blur_image = cv.blur(anterior_open_image, (5, 5), 5)
#     # 锐化
#     # k = np.array(([-1, 0, -1], [0, 5, 0], [-1, 0, -1]), np.float32)
#     # anterior_filter_image = cv.filter2D(anterior_blur_image, -1, kernel=k)
#
#     # images = np.hstack((anterior_image, anterior_gamma_image, anterior_binary_image, anterior_open_image, anterior_blur_image, anterior_filter_image))
#     # cv.imshow(str(id), images)
#     # cv.waitKey()
#
#     anterior_image_path = image_dictory + r'/GammaTransformedImages2/' + str(id) + 'a.jpg'
#     posterior_image_path = image_dictory + r'/GammaTransformedImages2/' + str(id) + 'p.jpg'
#
#     # cv.imwrite(anterior_image_path, anterior_binary_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
#     cv.imwrite(anterior_image_path, anterior_gamma_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
#     # cv.imwrite(posterior_image_path, posterior_binary_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])
#     cv.imwrite(posterior_image_path, posterior_gamma_image, [int(cv.IMWRITE_JPEG_QUALITY), 100])


def convertToRGB(root_path):
    all_image_list = os.listdir(IMAGE_DIR)
    image_set = set()
    for image_name in all_image_list:
        image_set.add(image_name[:-5])
    image_list = list(image_set)
    for image_name in image_list:
        print("Processing %s" % (image_name))
        ant_image_path = IMAGE_DIR + image_name + 'a.jpg'
        pos_image_path = IMAGE_DIR + image_name + 'p.jpg'
        ant_image = cv.imread(ant_image_path, cv.IMREAD_GRAYSCALE)
        pos_image = cv.imread(pos_image_path, cv.IMREAD_GRAYSCALE)
        images = np.hstack((ant_image, pos_image))
        # images_path = os.path.join(root_path, r'GammaTransformedImages/' + image_name + '.jpg')
        # images_hflip = cv2.flip(images, 1)
        # images_RGB = np.array([images, images_hflip, images])
        # images_RGB = np.moveaxis(images_RGB, 0, -1)
        # cv2.imwrite(images_path, images_RGB, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        images_path = os.path.join(root_path, r'GammaTransformedImages/' + image_name + '.jpg')
        cv.imwrite(images_path, images, [int(cv.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    images_list = os.listdir(IMAGE_DIR)
    gamma = 0.85
    gamma_images_save_dir = IMAGE_DIR + '-Gamma' + str(gamma)
    rgb_images_save_dir = IMAGE_DIR[:-4] + 'RGB-Gamma' + str(gamma)
    if not os.path.exists(gamma_images_save_dir):
        os.makedirs(gamma_images_save_dir)
    if not os.path.exists(rgb_images_save_dir):
        os.makedirs(rgb_images_save_dir)
    for image_name in images_list:
        print("Processing %s" % (image_name))
        image_path = os.path.join(IMAGE_DIR, image_name)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        gamma_image = gammaTransform(gamma, image, gamma_images_save_dir, image_name)
        image_hflip = cv.flip(gamma_image, 1)
        # image_RGB = np.array([gamma_image, gamma_image, gamma_image])
        image_RGB = np.array([gamma_image, image_hflip, gamma_image])
        image_RGB = np.moveaxis(image_RGB, 0, -1)

        rgb_image_path = os.path.join(rgb_images_save_dir, image_name)
        cv.imwrite(rgb_image_path, image_RGB, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    # convertToRGB(r'../../Resources/Dataset')
