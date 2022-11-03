import json
import os.path

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

ROOT_PATH = '../../Resources/Dataset/Annotations'
IMAGE_PATH = '../../Resources/Dataset/Images-Gray-Gamma1/16.jpg'
TRAIN_JSON = 'H-train.json'
TEST_JSON = 'H-test.json'


def json_parser(json_path):
    with open(json_path, 'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict['annotations']


def get_hotspots_list():
    hotspots_list = []
    train_json_path = os.path.join(ROOT_PATH, TRAIN_JSON)
    test_json_path = os.path.join(ROOT_PATH, TEST_JSON)
    train_dict = json_parser(train_json_path)
    test_dict = json_parser(test_json_path)
    for hotspot_dict in train_dict:
        bbox = hotspot_dict['bbox']
        x = bbox[0] + bbox[2] / 2
        y = bbox[1] + bbox[3] / 2
        hotspots_list.append([x, y])
    for hotspot_dict in test_dict:
        bbox = hotspot_dict['bbox']
        x = bbox[0] + bbox[2] / 2
        y = bbox[1] + bbox[3] / 2
        hotspots_list.append([x, y])

    return hotspots_list


def draw_point_on_image():
    hotspots_list = get_hotspots_list()
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    for hotspot in hotspots_list:
        x, y = hotspot
        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255))

    save_path = "../../Test/hotspots-statistics.jpg"
    cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # cv2.imshow('test', image)
    # cv2.waitKey(0)


def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


# 更新W
def update_W(X, Mu, Var, Pi):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return W


# 更新pi
def update_Pi(W):
    Pi = W.sum(axis=0) / W.sum()
    return Pi


# 计算log似然函数
def logLH(X, Pi, Mu, Var):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    return np.mean(np.log(pdfs.sum(axis=1)))


# 画出聚类图像
def plot_clusters(X, Mu, Var, Mu_true=None, Var_true=None):
    colors = ['b', 'g', 'r', 'b', 'g', 'r', 'b', 'g', 'r', 'b', 'g', 'r']
    n_clusters = len(Mu)
    plt.figure(figsize=(10, 8))
    plt.axis([0, 512, 0, 1024])
    # plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5)
    ax = plt.gca()
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        ellipse = Ellipse(Mu[i], 3 * Var[i][0], 3 * Var[i][1], **plot_args)
        ax.add_patch(ellipse)
    if (Mu_true is not None) & (Var_true is not None):
        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
            ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
            ax.add_patch(ellipse)
    plt.show()


# 更新Mu
def update_Mu(X, W):
    n_clusters = W.shape[1]
    Mu = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Mu[i] = np.average(X, axis=0, weights=W[:, i])
    return Mu


# 更新Var
def update_Var(X, Mu, W):
    n_clusters = W.shape[1]
    Var = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Var[i] = np.average((X - Mu[i]) ** 2, axis=0, weights=W[:, i])
    return Var


if __name__ == '__main__':
    # 生成数据
    # true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    # true_Var = [[1, 3], [2, 2], [6, 2]]
    # X = generate_X(true_Mu, true_Var)
    X = np.array(get_hotspots_list()[0:1000])
    # 初始化
    n_clusters = 2
    # n_clusters = 9
    n_points = len(X)
    # Mu = [[0, -1], [6, 0], [0, 9]]
    Mu = [[140, 400], [380, 400]]
    # Mu = [[120]]
    # Var = [[1, 1], [1, 1], [1, 1]]
    Var = [[80, 200], [80, 200]]
    Pi = [1 / n_clusters] * n_clusters
    W = np.ones((n_points, n_clusters)) / n_clusters
    Pi = W.sum(axis=0) / W.sum()
    # 迭代
    loglh = []
    for i in range(10):
        plot_clusters(X, Mu, Var)
        loglh.append(logLH(X, Pi, Mu, Var))
        W = update_W(X, Mu, Var, Pi)
        Pi = update_Pi(W)
        Mu = update_Mu(X, W)
        print('log-likehood:%.3f' % loglh[-1])
        Var = update_Var(X, Mu, W)
        print(Pi)
        print(Mu)
        print(Var)

