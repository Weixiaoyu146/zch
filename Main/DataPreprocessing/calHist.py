import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from pylab import *

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    n_path = r'../../Resources/Dataset/Images-Gray-Gamma1/185.jpg'
    m_path = r'../../Resources/Dataset/Images-Gray-Gamma1/177.jpg'
    a1_path = r'../../Resources/Dataset/Images-Gray-Gamma1/85.jpg'
    a2_path = r'../../Resources/Dataset/Images-Gray-Gamma1/79.jpg'

    n_image = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
    m_image = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
    a1_image = cv2.imread(a1_path, cv2.IMREAD_GRAYSCALE)
    a2_image = cv2.imread(a2_path, cv2.IMREAD_GRAYSCALE)

    # images = np.hstack((n_image, m_image, a1_image, a2_image))
    # cv2.imshow('1', images)
    # cv2.waitKey(0)

    # n_hist = cv2.calcHist([n_image], [0], None, [256], [0, 255])
    # m_hist = cv2.calcHist([m_image], [0], None, [256], [0, 255])
    # a1_hist = cv2.calcHist([a1_image], [0], None, [256], [0, 255])
    # a2_hist = cv2.calcHist([a2_image], [0], None, [256], [0, 255])
    #
    # n_sum = sum(n_hist[64:])
    # m_sum = sum(m_hist[64:])
    # a1_sum = sum(a1_hist[64:])
    # a2_sum = sum(a2_hist[64:])
    #
    # x_text = [r"正常图像", r"恶性肿瘤图像", r"手臂伪影图像", r"膀胱伪影图像"]
    # x_label = range(4)
    # y_label = [n_sum, m_sum, a1_sum, a2_sum]
    # col = ['g', 'r', 'y', 'b']
    #
    # plt.bar(x_label, y_label, color=col, alpha=0.75, width=0.5)
    # for a, b in zip(x_label, y_label):
    #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
    #
    # _ = plt.xticks(x_label, x_text)
    # plt.title(r"骨扫描图像批量归一化后灰度值≥64的像素点数")
    # plt.savefig('nums_pixel_64')
    # # plt.show()

    # x_text = ["1/36\n1/9"] * 13
    x_text = ["1/36\n2/9",
              "1/18\n2/9",
              "1/18\n4/9",
              "1/9\n4/9",
              "1/9\n8/9",
              "2/9\n8/9"]
    x_label = range(6)
    # y_label = [22] * 8
    y_label = [25.021, 23.893, 23.926,
               22.156, 21.641, 19.482]
    col = ['b'] * 6

    plt.ylim((18, 26))
    plt.xlabel('阈值组合（上为1/3个锚点框的阈值，下为3/5个锚点框的阈值）', fontsize=11)
    plt.ylabel('AP', fontsize=11)
    plt.bar(x_label, y_label, color=col, alpha=0.75, width=0.5)
    for a, b in zip(x_label, y_label):
        plt.text(a, b + 0.05, '%.03f' % b, ha='center', va='bottom', fontsize=11)

    _ = plt.xticks(x_label, x_text)
    plt.plot(x_label, y_label)
    plt.title(r"不同锚点框生成阈值组合实验（K=17）")
    plt.savefig('gmm-threshold', bbox_inches='tight')
    plt.show()
