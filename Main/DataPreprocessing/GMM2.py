import json
import os
from random import sample

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Ellipse
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy.linalg import det, inv
from sklearn.mixture import GaussianMixture

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


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=1)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=1)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def gaussion(x, mu, Sigma):
    dim = len(x)
    constant = (2 * np.pi) ** (-dim / 2) * det(Sigma) ** (-0.5)
    return constant * np.exp(-0.5 * (x - mu).dot(inv(Sigma)).dot(x - mu))


def gaussion_mixture(x, Pi, mu, Sigma):
    z = 0
    for idx in range(len(Pi)):
        z += Pi[idx] * gaussion(x, mu[idx], Sigma[idx]) * 128 * 256
    return z


if __name__ == '__main__':
    X = np.array(get_hotspots_list())
    # X = sample(get_hotspots_list(), 20000)
    # gmm = GaussianMixture(n_components=2, means_init=[[120, 400], [380, 400]]).fit(X)

    # gmm = GaussianMixture(n_components=4, means_init=[[120, 120],  # ?????????
    #                                                   [120, 450],  # ?????????
    #                                                   [380, 250],  # ?????????
    #                                                   [380, 450]]).fit(X)  # ?????????

    gmm = GaussianMixture(n_components=17, max_iter=2000, means_init=[[128, 128],  # ??????
                                                       [64, 170],  # ?????????
                                                       [192, 170],  # ?????????
                                                       [100, 300],  # ?????????
                                                       [156, 300],  # ?????????
                                                       [40, 280],  # ?????????
                                                       [200, 280],  # ?????????
                                                       [128, 430],  # ?????????
                                                       [108, 580],  # ?????????
                                                       [148, 580],  # ?????????
                                                       [380, 60],  # ??????
                                                       [300, 260],  # ?????????
                                                       [460, 280],  # ?????????
                                                       [380, 230],  # ?????????
                                                       [380, 435],  # ?????????
                                                       [350, 570],  # ?????????
                                                       [410, 590]]).fit(X)  # ?????????

    # labels = gmm.predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
    # plt.show()

    # plot_gmm(gmm, X)
    # plt.show()

    weights, means, covariances, = gmm.weights_, gmm.means_, gmm.covariances_

    x = np.linspace(0, 511, 128)
    y = np.linspace(0, 1023, 256)
    x, y = np.meshgrid(x, y)

    Z = np.array([x.ravel(), y.ravel()]).T
    z = [gaussion_mixture(i, weights, means, covariances) for i in Z]
    z = np.array(z).reshape(x.shape)

    # fig = plt.figure()
    # # ??????3d??????
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax1.plot_surface(x, y, z)
    #
    # # ???????????????
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.contour(x, y, z)

    fig = plt.figure()
    # ????????????????????? 3d ??????
    ax = plt.axes(projection='3d')
    # ax = fig.add_subplot(1, 2, 1)
    # Plot the surface.

    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_inverted(True)
    ax.zaxis.set_major_locator(LinearLocator(11))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.plot_surface(x, y, z)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # ax.contour(x, y, z)

    plt.show()
