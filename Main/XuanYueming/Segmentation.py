import os
import cv2 as cv
from Region_growing import *
from ReadXml import *
from Preprocessing import image_enhance


class Segmentation:
    a = b = c = d = e = f = g = h = i = j = k = l = m = n = o = p = q = r = s = t = u = v = (0, 0)
    head_left = head_right = foot_joint_row = foot_row = left_hand_row = right_hand_row = 0
    vertbra_contour = []
    vertbra = []
    kidneys = []
    bladder = []
    imageA = imageP = []  # 预处理后分割用图像
    height = width = 0  # 图像高宽

    histogram_A = \
        histogram_P = []  # 前身图直方图列表 #后身图直方图列表
    OriginA = ''  # 前身图
    OriginP = ''  # 后身图
    nameA = ''  # 前身图名字
    nameP = ''  # 后身图名字
    imgs_vertbra = []  # 脊椎部位图
    imgs_pelvis = []  # 骨盆部位图
    labels_vertbra = []  # 脊椎前后身标签
    labels_pelvis = []  # 骨盆前后身标签

    # 类的实例化
    def __init__(self, imageA, imageP, OriginA, OriginP, nameA='', nameP=''):
        if len(imageA.shape) != 2:
            print("error image struct")
            return
        self.histogram_A = np.sum(imageA, axis=1) / imageA.shape[1]
        self.histogram_P = np.sum(imageP, axis=1) / imageP.shape[1]
        # print(self.histogram_P)

        self.OriginA = OriginA
        self.OriginP = OriginP
        self.nameA = nameA
        self.nameP = nameP
        self.imageA = imageA
        self.imageP = imageP
        self.height, self.width = imageA.shape

    def Get_points(self):
        # 点的定位以及脊柱的定位和肾脏的定位
        self.get_ab()
        self.get_ij()
        self.get_cdefgh()
        self.get_kl()
        self.get_mn()
        self.get_opst()
        self.get_uv()
        self.get_joint()
        self.get_vertbra_contour()  # 脊椎轮廓
        self.get_vertbra_kidneys()  # 肾脏

        return

    # 分割提取脊椎部位
    def Segment_vertbra(self):

        vA, vP = self.cut_vertbra()  # 获取分割图像和标签

        tempsV = self.Cut_Region(vA, vP)  # 分割图像剪切
        #
        # self.imgs_vertbra = self.Image_Resize(tempsV, 0)  # 分割图像增强并调整尺寸
        #
        # self.Save_Image('./seg_test/vertbra/', self.imgs_vertbra)  # 保存部位图像

        # 绘制定位点和框
        # self.show_point()
        cv.imshow('segP', self.show_segment())
        # cv.waitKey(10)

        return

    # 分割提取骨盆部位
    def Segment_pelvis(self):

        pA, pP, self.labels_pelvis = self.cut_pelvis()  # 获取分割图像和标签

        tempsP = self.Cut_Region(pA, pP)  # 分割图像剪切

        self.imgs_pelvis = self.Image_Resize(tempsP, 1)  # 分割图像增强并调整尺寸

        self.Save_Image('./seg_test/pelvis/', self.imgs_pelvis)  # 保存部位图像

        # 绘制定位点和框
        # self.show_point()
        # cv.imshow('segP', self.show_segment())
        # cv.waitKey(10)

        return

    # 绘制点和划线
    def show_point(self):
        imageP = self.imageP
        imageA = self.imageA
        imageP = cv.cvtColor(imageP, cv.COLOR_GRAY2RGB)
        imageA = cv.cvtColor(imageA, cv.COLOR_GRAY2RGB)
        cv.circle(img=imageP, center=self.a, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.b, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.c, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.d, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.e, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.f, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.g, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.h, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.i, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.j, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.k, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.l, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.m, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=self.n, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageP, center=(self.width - self.o[0] - 1, self.o[1]), radius=3, color=(0, 255, 0),
                  thickness=3)
        cv.circle(img=imageP, center=(self.width - self.p[0] - 1, self.p[1]), radius=3, color=(0, 255, 0),
                  thickness=3)
        cv.circle(img=imageP, center=(self.width - self.s[0] - 1, self.s[1]), radius=3, color=(0, 255, 0),
                  thickness=3)
        cv.circle(img=imageP, center=(self.width - self.t[0] - 1, self.t[1]), radius=3, color=(0, 255, 0),
                  thickness=3)

        cv.circle(img=imageA, center=self.o, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageA, center=self.p, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageA, center=self.s, radius=3, color=(0, 0, 255), thickness=3)
        cv.circle(img=imageA, center=self.t, radius=3, color=(0, 0, 255), thickness=3)

        cv.circle(img=imageP, center=self.u, radius=3, color=(255, 0, 0), thickness=3)
        cv.circle(img=imageP, center=self.v, radius=3, color=(255, 0, 0), thickness=3)
        cv.circle(img=imageP, center=(self.c[0], min(self.g[1], self.h[1])), radius=3, color=(255, 0, 0), thickness=3)
        cv.circle(img=imageP, center=(self.d[0], min(self.g[1], self.h[1])), radius=3, color=(255, 0, 0), thickness=3)

        cv.imshow(self.nameP, imageP)

    # 1.绘制方框 2.分割部位分别保存
    def show_segment(self):
        # 1.绘制边框直线
        imageP = image_enhance(self.OriginP)
        imageP = cv.cvtColor(imageP, cv.COLOR_GRAY2RGB)
        red = (0, 0, 255)
        green = (0, 255, 0)

        # for point in self.vertbra:
        #     imageP[point[1], point[0]] = red

        # 脊椎
        cv.line(imageP, (self.c[0], min(self.g[1], self.h[1])), (self.d[0], min(self.g[1], self.h[1])), green, 1)
        cv.line(imageP, self.u, self.v, green, 1)
        cv.line(imageP, (self.c[0], min(self.g[1], self.h[1])), self.u, green, 1)
        cv.line(imageP, (self.d[0], min(self.g[1], self.h[1])), self.v, green, 1)

        # 头部
        cv.rectangle(imageP, (self.head_left, 0), (self.head_right, self.a[1]), red, 1)
        # 左肩
        cv.line(imageP, (0, self.e[1]), self.e, red, 1)
        cv.line(imageP, (0, self.g[1]), self.g, red, 1)
        cv.line(imageP, self.e, self.c, red, 1)
        cv.line(imageP, self.g, self.c, red, 1)
        # 右肩
        cv.line(imageP, (self.width - 1, self.f[1]), self.f, red, 1)
        cv.line(imageP, (self.width - 1, self.h[1]), self.h, red, 1)
        cv.line(imageP, self.f, self.d, red, 1)
        cv.line(imageP, self.d, self.h, red, 1)

        # 左右手
        cv.line(imageP, self.g, (self.g[0], self.m[1]), red, 1)
        cv.line(imageP, (self.g[0], self.m[1]), (0, self.m[1]), red, 1)

        cv.line(imageP, self.h, (self.h[0], self.n[1]), red, 1)
        cv.line(imageP, (self.width - 1, self.n[1]), (self.h[0], self.n[1]), red, 1)
        # 左右手关节
        cv.rectangle(imageP, (0, self.left_hand_row - 20), (self.g[0], self.left_hand_row + 20), red, 1)
        cv.rectangle(imageP, (self.h[0], self.right_hand_row - 20), (self.width - 1, self.right_hand_row + 20),
                     red, 1)

        # # 盆骨
        cv.rectangle(imageP, (self.width - self.o[0] - 1, self.i[1]),
                     (self.width - self.p[0] - 1, self.n[1]), red, 1)

        # 左右腿
        cv.rectangle(imageP, (self.e[0], self.m[1]), (int((self.e[0] + self.f[0]) / 2), self.height - 1), red, 1)
        cv.rectangle(imageP, (int((self.e[0] + self.f[0]) / 2), self.m[1]), (self.f[0], self.height - 1), red, 1)

        # 左右腿关节
        cv.rectangle(imageP, (self.e[0], self.foot_joint_row - 25),
                     (int((self.e[0] + self.f[0]) / 2), self.foot_joint_row + 25), red, 1)
        cv.rectangle(imageP, (int((self.e[0] + self.f[0]) / 2), self.foot_joint_row - 25),
                     (self.f[0], self.foot_joint_row + 25), red, 1)

        # 左右脚掌
        cv.rectangle(imageP, (self.e[0], self.foot_row - 15), (int((self.e[0] + self.f[0]) / 2), self.height),
                     red, 1)
        cv.rectangle(imageP, (int((self.e[0] + self.f[0]) / 2), self.foot_row - 15), (self.f[0], self.height),
                     red, 1)

        return imageP

    # 获得图像中特定范围内的行的第一个非0元素离边框距离
    # params:
    # Y_1:搜索范围的起始行
    # Y_2:搜索范围的终止行
    # X_start:每一行从X_start列开始搜索直到列尾
    # X_div_start:每一行从右边第X_div_start列开始搜索直到列头
    # return 特定范围内的行的第一个非0元素离边框距离
    def get_first_nozeros(self, image, Y_1, Y_2, X_start=0, X_div_start=0):
        first_nozeros = []
        div_first_nozeros = []
        for row in range(image.shape[0]):
            first_nozeros.append(0)
            # first_nozeros[row] = imageP.shape[1]-1 #搜索非0元素所在列前，将其初始化为最右边的点
            div_first_nozeros.append(0)
            # div_first_nozeros[row] = imageP.shape[1]-1
        for row in range(Y_1, Y_2):
            for col in range(X_start, image.shape[1]):
                if image[row][col] != 0:
                    first_nozeros[row] = col
                    break
            for col in range(X_div_start, image.shape[1]):
                if image[row][image.shape[1] - col - 1] != 0:
                    div_first_nozeros[row] = col
                    break
        return first_nozeros, div_first_nozeros

    # 定位ab两点以及头部位置
    def get_ab(self):
        Y_10 = int(0.10 * self.height)
        Y_20 = int(0.20 * self.height)
        min = 256
        index = 0
        X_left = 0
        X_right = 0
        # 寻找ab所在行
        for row in range(Y_10, Y_20):
            if self.histogram_P[row] < min:
                min = self.histogram_P[row]
                index = row
        # 寻找ab所在列
        for col in range(self.width):
            if self.imageP[index][col] != 0:
                X_left = col
                break
        for col in range(self.width):
            if self.imageP[index][self.width - col - 1] != 0:
                X_right = self.width - col - 1
                break
        self.a = (X_left, index)
        self.b = (X_right, index)
        if abs(X_right - X_left) == 0:
            self.b = (X_right + 10, index)
            if self.a[0] > 10:
                self.a = (X_left - 10, index)
        head_left = self.width - 1
        head_right = self.width - 1

        # 大脑左右边界无法通过特征点来确定，需要初始定位：
        first_nozeros, div_first_nozeros = self.get_first_nozeros(image=self.imageP, Y_1=0, Y_2=self.a[1])
        for row in range(self.a[1]):
            if first_nozeros[row] < head_left and first_nozeros[row] != 0:
                head_left = first_nozeros[row]

        min = self.width - 1
        for row in range(self.a[1]):
            if div_first_nozeros[row] < min and div_first_nozeros[row] != 0:
                min = div_first_nozeros[row]
                head_right = self.width - div_first_nozeros[row] - 1
        self.head_left = head_left
        self.head_right = head_right

    # 定位i,j两点位置并依据ab位置修正
    def get_ij(self):
        Y_30 = int(0.30 * self.height)
        Y_45 = int(0.45 * self.height)
        min = 256
        index = 0
        X_left = 0
        X_right = 0
        # 寻找ij所在行
        for row in range(Y_30, Y_45):
            if self.histogram_P[row] < min:
                min = self.histogram_P[row]
                index = row
        # 寻找ij所在列
        for col in range(int(self.width / 4), self.width):
            if self.imageP[index][col] != 0:
                X_left = col
                break
        for col in range(int(self.width / 4), self.width):
            if self.imageP[index][self.width - col - 1] != 0:
                X_right = self.width - col - 1
                break
        self.i = (X_left, index)
        self.j = (X_right, index)
        # i,j两点X坐标修正：
        X_ab = float(self.b[0] - self.a[0])
        X_ij = float(self.j[0] - self.i[0])
        X_ai = float(self.a[0] - self.i[0])
        X_bj = float(self.b[0] - self.j[0])
        if X_ij / X_ab > 1.2:
            if abs(X_ai) - abs(X_bj) > 15:
                X_left = self.a[0] + self.j[0] - self.b[0]
                self.i = (X_left, index)
            if abs(X_bj) - abs(X_ai) > 15:
                X_right = self.b[0] + self.i[0] - self.a[0]
                self.j = (X_right, index)

    # 定位c,d,e,f,g,h位置
    def get_cdefgh(self):
        # cdefgh定位坐标参数：
        Y_ce = 0
        Y_g = 0
        Y_df = 0
        Y_h = 0
        X_eg = 0
        X_c = 0
        X_fh = 0
        X_d = 0
        Y_1 = self.a[1]
        Y_2 = int(0.20 * self.height)
        # 求出每一行第一个非0元素距离边框的位置（方式为从左往右计数和从右往左计数）
        first_nozeros, div_first_nozeros = self.get_first_nozeros(self.imageP, Y_1, Y_2)
        # 计算相关坐标
        pre_nozeros = first_nozeros[Y_1]
        div_pre_nozeros = div_first_nozeros[Y_1]
        for row in range(Y_1, Y_2):
            now_nozeros = first_nozeros[row]
            if now_nozeros == 0:
                now_nozeros += 1
            if float(pre_nozeros / now_nozeros) > 1.5:  # 如果往下搜索过程中非0行所在列发生突变，认为找到肩部
                Y_ce = row
                X_eg = first_nozeros[row]
                break
            else:
                pre_nozeros = now_nozeros
        for row in range(Y_1, Y_2):
            div_now_nozeros = div_first_nozeros[row]
            if div_now_nozeros == 0:
                div_now_nozeros += 1
            if float(div_pre_nozeros / div_now_nozeros) > 1.5:
                Y_df = row
                X_fh = self.width - div_first_nozeros[row] - 1
                break
            else:
                div_pre_nozeros = div_now_nozeros
        '''
        #计算cd两点所在列?????
        for col in range(X_eg+10, self.width):
            if self.imageP[Y_ce][col] != 0:
                X_c = col
                break
        for col in range(self.width-X_fh+10, self.width-1):
            if self.imageP[Y_df][self.width-col-1] != 0:
                X_d = self.width-col-1
                break
        '''
        # 计算gh两点所在行
        for row in range(Y_ce, Y_2):
            if self.imageP[row][X_eg] != 0:
                if self.imageP[row + 1][X_eg] != 0 and self.imageP[row + 2][X_eg] != 0:
                    Y_g = row

        for row in range(Y_df, Y_2):
            if self.imageP[row][X_fh] != 0:
                if self.imageP[row + 1][X_fh] != 0 and self.imageP[row + 2][X_fh] != 0:
                    Y_h = row

        self.c = (self.a[0], Y_ce)
        self.d = (self.b[0], Y_df)
        self.e = (X_eg, Y_ce)
        self.f = (X_fh, Y_df)
        self.g = (X_eg, Y_g + 10)
        self.h = (X_fh, Y_h + 10)
        # 重定位：
        '''
        X_5 = int(0.05 * self.width)
        X_50 = int(0.5 * self.width)
        X_95 = int(0.95 * self.width)
        x = 0
        min = 9999
        if X_eg <= X_5 or Y_ce < self.a[1]:
            Col_hist = np.sum(self.imageP[:, X_5:X_50], axis=0) / self.imageP.shape[0]
            # print(Col_hist)
            # print(len(Col_hist))
            for i in range(len(Col_hist)):
                if Col_hist[i] <= min:
                    min = Col_hist[i]
                    x = i
            X_eg = x + X_5
            print('e', X_eg)

        for Y in range(self.height):
            if self.imageP[Y][X_eg] != 0:
                Y_ce = Y
                break
        if X_fh <= self.b[0] or Y_df < self.b[1]:
            Col_hist = np.sum(self.imageP[:, X_50:X_95], axis=0) / self.imageP.shape[0]
            # print(Col_hist)
            # print(len(Col_hist))
            for i in range(len(Col_hist)):
                if Col_hist[i] <= min:
                    min = Col_hist[i]
                    x = i
            X_fh = x + X_5
            print('f', X_fh)

        for Y in range(self.height):
            if self.imageP[Y][X_fh] != 0:
                Y_df = Y
                break
        '''

        Col_hist = np.sum(self.imageP, axis=0) / self.imageP.shape[0]
        Middle_axis = int((self.a[0] + self.b[0]) / 2)

        # 左肩正常右肩异常
        if Y_ce >= self.a[1] and (Y_df < self.b[1] or X_fh <= self.b[0]):
            X_right = 2 * Middle_axis - self.e[0]
            min = 255
            X_fh = 0
            range_right = np.minimum(X_right + 20, self.width - 1)

            for col in range(X_right - 20, range_right):
                if Col_hist[col] < min and Col_hist[col] != 0:
                    min = Col_hist[col]
                    X_fh = col - 10
            for row in range(self.b[1], self.b[1] + 40):
                if self.imageP[row][X_fh] != 0:
                    Y_df = row
                    break
            for row in range(Y_df, Y_2):
                if self.imageP[row][X_fh] != 0:
                    if self.imageP[row + 1][X_fh] != 0 and self.imageP[row + 2][X_fh] != 0:
                        Y_h = row
            self.d = (self.b[0], Y_df)
            self.f = (X_fh, Y_df)
            self.h = (X_fh, Y_h + 10)

        # 左肩异常右肩正常：
        if Y_ce < self.a[1] and Y_df >= self.b[1]:
            X_left = 2 * Middle_axis - self.f[0]
            min = 255
            X_eg = 0
            range_left = np.maximum(X_left - 20, 0)
            for col in range(range_left, X_left + 20):
                if Col_hist[col] < min and Col_hist[col] != 0:
                    min = Col_hist[col]
                    X_eg = col + 10
            for row in range(self.a[1], self.a[1] + 40):
                if self.imageP[row][X_eg] != 0:
                    Y_ce = row
                    break
            for row in range(Y_ce, Y_2):
                if self.imageP[row][X_eg] != 0:
                    if self.imageP[row + 1][X_eg] != 0 and self.imageP[row + 2][X_eg] != 0:
                        Y_g = row
            self.c = (self.a[0], Y_ce)
            self.e = (X_eg, Y_ce)
            self.g = (X_eg, Y_g + 10)

        # 肩高过小
        if self.g[1] - self.e[1] < 30:
            self.g = (self.e[0], self.e[1] + 30)
        if self.h[1] - self.f[1] < 30:
            self.h = (self.f[0], self.f[1] + 30)

    def get_kl(self):
        Y_k = 0
        Y_l = 0
        X_k = 0
        X_l = 0
        first_nozeros, div_first_nozeros = self.get_first_nozeros(image=self.imageP, Y_1=self.i[1], Y_2=self.i[1] + 30,
                                                                  X_start=self.e[0],
                                                                  X_div_start=self.width - self.f[0] - 1)
        Y_1 = self.i[1]  # 注意Y_l与Y_1不是一个变量
        Y_2 = self.i[1] + 30
        pre_nozeros = first_nozeros[Y_1]
        div_pre_nozeros = div_first_nozeros[Y_1]
        for row in range(Y_1, Y_2):
            now_nozeros = first_nozeros[row]
            if now_nozeros == 0:
                now_nozeros += 1
            if pre_nozeros - now_nozeros > 11:  # 如果往下搜索过程中非0行所在列发生突变，认为找到盆骨上端
                Y_k = row
                X_k = first_nozeros[row]
                break
            else:
                pre_nozeros = now_nozeros
        for row in range(Y_1, Y_2):
            div_now_nozeros = div_first_nozeros[row]
            if div_now_nozeros == 0:
                div_now_nozeros += 1
            if div_pre_nozeros - div_now_nozeros > 11:
                Y_l = row
                X_l = self.width - div_first_nozeros[row] - 1
                break
            else:
                div_pre_nozeros = div_now_nozeros

        self.k = (X_k, Y_k)
        self.l = (X_l, Y_l)
        # k l 点位置修正
        # k,l都出现问题：
        if X_l == X_k == Y_l == Y_k == 0:
            self.k = (self.i[0] - 15, min(self.i[1] + 10, self.height - 1))
            self.l = (min(self.j[0] + 15, self.width - 1), min(self.j[1] + 10, self.height - 1))
        # l正常，k出现问题：
        if (X_k == 0 or Y_k == 0) and X_l != 0 and Y_l != 0:
            self.k = (min(self.i[0] + self.j[0] - self.l[0], self.width - 1), Y_1)
        # k正常，l出现问题：
        if (X_l == 0 or Y_l == 0) and X_k != 0 and Y_k != 0:
            self.l = (min(self.j[0] + self.i[0] - self.k[0], self.width - 1), Y_k)

        # print("...............")
        # print(first_nozeros)
        # print(div_first_nozeros)

    def get_mn(self):
        Y_11 = int(self.height * 0.11)
        Y_16 = int(self.height * 0.16)
        Y_m = 0
        Y_n = 0

        list = []
        # 从k点往下搜索全身的11%到16%处就行
        for row in range(self.k[1] + Y_11, self.k[1] + Y_16):
            Flag = True
            for col in range(self.k[0], self.k[0] + 10):
                if self.imageP[row][col] != 0:
                    Flag = False
                    break
            if Flag == True:
                Y_m = row
                break

        for row in range(self.l[1] + Y_11, self.l[1] + Y_16):
            Flag = True
            for col in range(self.l[0] - 10, self.l[0]):
                if self.imageP[row][col] != 0:
                    Flag = False
                    break
            if Flag == True:
                Y_n = row
                break
        self.m = (self.k[0], Y_m)
        self.n = (self.l[0], Y_n)
        # 重定位
        # 左边正常右边异常
        if Y_m > self.k[1] and Y_n <= self.l[1]:
            Y_n = Y_m
            self.n = (self.l[0], Y_n)
        # 右边正常左边异常
        if Y_m <= self.k[1] and Y_n > self.l[1]:
            Y_m = Y_n
            self.m = (self.k[0], Y_m)
        # 两边都异常
        if Y_m <= self.k[1] and Y_n <= self.l[1]:
            Y_n = Y_m = self.k[1] + int(self.height * 0.12)
            self.m = (self.k[0], Y_m)
            self.n = (self.l[0], Y_n)
        self.m = (self.k[0], int((Y_m + Y_n) / 2))
        self.n = (self.l[0], int((Y_m + Y_n) / 2))

    def get_opst(self):
        first_nozeros, div_first_nozeros = self.get_first_nozeros(image=self.imageA, Y_1=self.i[1], Y_2=self.i[1] + 50,
                                                                  X_div_start=self.e[0],
                                                                  X_start=self.width - self.f[0] - 1)
        Y_1 = self.i[1]
        Y_2 = self.i[1] + 50
        min = self.width
        row_index = 0
        for row in range(Y_1, Y_2):
            if first_nozeros[row] < min:
                min = first_nozeros[row]
                row_index = row
        self.o = (min, row_index)

        min = self.width
        row_index = 0
        for row in range(Y_1, Y_2):
            if div_first_nozeros[row] < min:
                min = div_first_nozeros[row]
                row_index = row

        self.p = (self.width - min, row_index)
        self.s = (self.o[0] - 10, self.o[1] - 20)
        self.t = (self.p[0] + 10, self.p[1] - 20)

    # 得到手关节，脚关节，手掌，脚掌位置并返回结果。

    def get_uv(self):
        Y_min = max(self.i[1], self.j[1]) + 50
        Y_max = self.m[1] - 30
        min = 256
        index = 0
        for row in range(Y_min, Y_max):
            if self.histogram_A[row] < min:
                min = self.histogram_A[row]
                index = row
        index -= 10
        self.u = (self.i[0], index)
        self.v = (self.j[0], index)

    def get_joint(self):
        # 左手关节位置:
        data = self.imageP[:, 0:self.g[0] - 5]
        exist = (data > 0) * 1.0
        left_hand = np.sum(exist, axis=1)
        max = 0
        left_hand_row = 0
        for row in range(self.i[1] - 50, self.i[1]):
            if left_hand[row] + left_hand[row - 1] + left_hand[row + 1] > max:
                max = left_hand[row] + left_hand[row - 1] + left_hand[row + 1]
                left_hand_row = row
        # 右手关节位置
        data = self.imageP[:, self.h[0] + 5:self.width - 1]
        exist = (data > 0) * 1.0
        right_hand = np.sum(exist, axis=1)
        max = 0
        right_hand_row = 0
        for row in range(self.i[1] - 50, self.i[1]):
            if right_hand[row] + right_hand[row - 1] + right_hand[row + 1] > max:
                max = right_hand[row] + right_hand[row - 1] + right_hand[row + 1]
                right_hand_row = row

        # 脚关节位置:
        exist = (self.imageP > 0) * 1.0
        foot = np.sum(exist, axis=1)
        max = 0
        foot_joint_row = 0
        for row in range(self.m[1] + 50, self.height - 100):
            if foot[row] + foot[row - 1] + foot[row + 1] > max:
                max = foot[row] + foot[row - 1] + foot[row + 1]
                foot_joint_row = row
        if foot_joint_row == 0:
            foot_joint_row = int((self.m[1] + self.height - 1) / 2)

        # 脚掌定位:
        max = 0
        foot_row = 0
        for row in range(self.height - 100, self.height - 1):
            if foot[row] + foot[row - 1] + foot[row + 1] > max:
                max = foot[row] + foot[row - 1] + foot[row + 1]
                foot_row = row
        if foot_row == 0:  # 重定位
            foot_row = self.height - 80

        self.foot_row = foot_row
        self.foot_joint_row = foot_joint_row
        self.left_hand_row = left_hand_row
        self.right_hand_row = right_hand_row

    def takeFirst(self, elem):
        return elem[0]

    # 列表按Y排序
    def Correction_information(self, vertbra):
        Correction_information = []  # 脊柱修正信息

        def takeSecond(elem):
            return elem[1]

        vertbra_sort = vertbra.copy()
        vertbra_sort.sort(key=takeSecond)
        temp_list = []
        # 信息处理1.排序显示
        for point in vertbra_sort:
            if len(temp_list):
                if temp_list[0][1] == point[1]:
                    temp_list.append(point)

                else:
                    temp_list.sort(key=self.takeFirst)
                    Correction_information.append(temp_list.copy())
                    temp_list.clear()
                    temp_list.append(point)
            else:
                temp_list.append(point)
        temp_list.sort(key=self.takeFirst)
        Correction_information.append(temp_list.copy())
        return Correction_information

    def get_points_list(self, pointA, pointB):
        list = []
        if pointA[0] == pointB[0]:
            for y in range(min(pointA[1], pointB[1]), max(pointA[1], pointB[1]) + 1):
                list.append((pointA[0], y))
            return list
        if pointA[1] == pointB[1]:
            for x in range(min(pointA[0], pointB[0]), max(pointA[0], pointB[0]) + 1):
                list.append((x, pointA[1]))
            return list

        # 如果斜率大于1的话：
        if abs(pointA[1] - pointB[1]) >= abs(pointA[0] - pointB[0]):
            if pointA[1] < pointB[1]:
                k = (pointA[0] - pointB[0]) / (pointA[1] - pointB[1])  # 斜率公式
                for y in range(pointA[1], pointB[1] + 1):
                    list.append((int((y - pointB[1]) * k + pointB[0]), y))
            if pointA[1] > pointB[1]:
                k = (pointA[0] - pointB[0]) / (pointA[1] - pointB[1])  # 斜率公式
                for y in range(pointB[1], pointA[1] + 1):
                    list.append((int((y - pointB[1]) * k + pointB[0]), y))
            return list

        # # 如果斜率小于1的话：
        # if abs(pointA[1] - pointB[1]) <= abs(pointA[0] - pointB[0]):
        #     k = (pointA[1] - pointB[1]) / (pointA[0] - pointB[0])  # 斜率公式
        #     if pointA[1] < pointB[1]:
        #         for y in range(pointA[1], pointB[1] + 1):
        #             list.append((int((y - pointB[1]) * k + pointB[0]), y))
        #     if pointA[1] > pointB[1]:
        #         for y in range(pointB[1], pointA[1] + 1):
        #             list.append((int((y - pointB[1]) * k + pointB[0]), y))
        #     return list

    # 线条闭合曲线拟合
    def get_connect(self, down, up, vertbra):
        length = len(vertbra)
        X_bias = vertbra[down][0] - vertbra[up][0]
        Y_bias = vertbra[down][1] - vertbra[up][1]
        list = []
        if X_bias < 0:
            for index in range(vertbra[down][0], vertbra[up][0] - 1):
                list.append((index + 1, vertbra[down][1]))
            if Y_bias < 0:
                for index in range(vertbra[down][1], vertbra[up][1] + 1):
                    list.append((vertbra[up][0], index))
            if Y_bias > 0:
                for index in range(vertbra[down][1], vertbra[up][1] - 1, -1):
                    list.append((vertbra[up][0], index))
        if X_bias == 0:
            if Y_bias < 0:
                for index in range(vertbra[down][1], vertbra[up][1] + 1):
                    list.append((vertbra[up][0], index))
            if Y_bias > 0:
                for index in range(vertbra[down][1], vertbra[up][1] - 1, -1):
                    list.append((vertbra[up][0], index))
        if X_bias > 0:
            for index in range(vertbra[down][0], vertbra[up][0] + 1, -1):
                list.append((index - 1, vertbra[down][1]))
            if Y_bias < 0:
                for index in range(vertbra[down][1], vertbra[up][1] + 1):
                    list.append((vertbra[up][0], index))
            if Y_bias > 0:
                for index in range(vertbra[down][1], vertbra[up][1] - 1, -1):
                    list.append((vertbra[up][0], index))

        if up < down:
            del vertbra[down + 1:length]
            vertbra.extend(list)
            del vertbra[0:up]
        if down < up:
            temp = vertbra[up + 1:length]
            del vertbra[down + 1:length]
            vertbra.extend(list)
            vertbra.extend(temp)

        return vertbra

    # 获取脊柱轮廓并修正
    def get_vertbra_contour(self):
        imageP = self.imageP.copy()
        cv.threshold(imageP, 230, 255, cv.THRESH_TOZERO, imageP)  # 简单阈值分割
        contours, hierarchy = cv.findContours(imageP, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # 脊柱轮廓粗提取
        max_area = 0
        max_contour = contours[0]
        for contour in contours:
            area = cv.contourArea(contour)
            if area > max_area:
                max_contour = contour
                max_area = area

        x, y, w, h = cv.boundingRect(max_contour)
        # 轮廓信息转变为列表存储格式：
        max_contour.tolist()
        vertbra = []

        for points in max_contour:
            if points[0][1] >= self.a[1] and points[0][1] <= self.j[1]:
                vertbra.append((points[0][0], points[0][1]))

        # 轮廓像素按升序排序
        vertbra_sort = vertbra.copy()

        vertbra_sort.sort(key=self.takeFirst)

        # 修正脊柱信息：
        # 没找到脊柱（列表为空）:
        if len(vertbra_sort) == 0:
            list_line = self.get_points_list(self.a, self.i)
            list_line2 = self.get_points_list(self.b, self.j)
            list_line.extend(list_line2)
            vertbra = list_line.copy()
        else:
            # 左边像素越界太多
            while (self.a[0] + self.i[0]) / 2 - vertbra_sort[0][0] > 10:
                index = vertbra.index(vertbra_sort[0])
                up = down = index
                length = len(vertbra)
                Y_height = vertbra_sort[0][1]
                while vertbra[up][0] < (self.a[0] + self.i[0]) / 2 - 5 or (
                        vertbra[up][1] < Y_height + 20 and vertbra[up][1] > Y_height - 20):
                    up += 1
                    if up == length:
                        up = 0

                while vertbra[down][0] < (self.a[0] + self.i[0]) / 2 - 5 or (
                        vertbra[down][1] < Y_height + 20 and vertbra[down][1] > Y_height - 20):
                    down -= 1
                    if down == -1:
                        down = length - 1

                vertbra = self.get_connect(down, up, vertbra)

                correction_information = self.Correction_information(vertbra)
                vertbra_sort = vertbra.copy()
                vertbra_sort.sort(key=self.takeFirst)
            # 右边像素越界太多
            while vertbra_sort[len(vertbra) - 1][0] - (self.b[0] + self.j[0]) / 2 > 10:
                index = vertbra.index(vertbra_sort[len(vertbra) - 1])
                up = down = index
                Y_height = vertbra_sort[len(vertbra) - 1][1]
                length = len(vertbra)
                while vertbra[up][0] - (self.b[0] + self.j[0]) / 2 - 5 > 0 or (
                        vertbra[up][1] < Y_height + 20 and vertbra[up][1] > Y_height - 20):
                    up += 1
                    if up == length:
                        up = 0
                while vertbra[down][0] - (self.b[0] + self.j[0]) / 2 - 5 > 0 or (
                        vertbra[down][1] < Y_height + 20 and vertbra[down][1] > Y_height - 20):
                    down -= 1
                    if down == -1:
                        down = length - 1
                vertbra = self.get_connect(down, up, vertbra)

                vertbra_sort = vertbra.copy()
                vertbra_sort.sort(key=self.takeFirst)

        # 是否完全舍弃脊柱定位选择重置:

        if y > self.a[1] + 15:
            list_line = self.get_points_list(self.a, self.i)
            list_line2 = self.get_points_list(self.b, self.j)
            list_line.extend(list_line2)
            vertbra = list_line.copy()

        self.vertbra_contour = vertbra

    def get_vertbra_kidneys(self):
        image = self.OriginP.copy()
        vertbra_Y = self.Correction_information(self.vertbra_contour)
        black = (0, 0, 0)
        kidneys = []
        vertbra = []
        for list in vertbra_Y:

            for x in range(list[0][0], list[len(list) - 1][0]):
                image[list[0][1]][x] = 0
                vertbra.append((x, list[0][1]))

        # 将胸腔分成4个区域：

        image = cv.medianBlur(image, 3)

        area_1 = image[self.e[1]:int((self.e[1] + self.i[1]) / 2), self.e[0]:self.a[0]]
        area_2 = image[self.f[1]:int((self.f[1] + self.j[1]) / 2), self.b[0]:self.f[0]]
        area_3 = image[int((self.e[1] + self.i[1]) / 2):self.i[1], self.e[0]:self.a[0]]
        area_4 = image[int((self.f[1] + self.j[1]) / 2):self.j[1], self.b[0]:self.f[0]]
        regionGrow_3 = []
        regionGrow_4 = []

        if cv.mean(area_1)[0] < cv.mean(area_3)[0] * 1.2:

            area_3_detect = area_3[int(area_3.shape[0] * 0.5):int(area_3.shape[0] * 0.8),
                            int(area_3.shape[1] * 0.5):int(area_3.shape[1])]

            _, max_3, _, max_index_3 = cv.minMaxLoc(area_3_detect)

            seeds_3 = [Point(max_index_3[1] + int(area_3.shape[0] * 0.5), max_index_3[0] + int(area_3.shape[1] * 0.5))]

            regionGrow_3 = regionGrow(area_3, seeds_3, max_3)

            for point in regionGrow_3:
                area_3[point[1], point[0]] = 0
                image[point[1] + int((self.e[1] + self.i[1]) / 2), point[0] + self.e[0]] = 0
                kidneys.append((point[0] + self.e[0], point[1] + int((self.e[1] + self.i[1]) / 2)))

            # cv.imshow("3", area_3)
            # cv.imshow("4", area_4)
        if cv.mean(area_2)[0] < cv.mean(area_4)[0] * 1.2:
            area_4_detect = area_4[int(area_4.shape[0] * 0.5):int(area_4.shape[0] * 0.8), :int(area_4.shape[1] * 0.5)]
            _, max_4, _, max_index_4 = cv.minMaxLoc(area_4_detect)
            seeds_4 = [Point(max_index_4[1] + int(area_4.shape[0] * 0.5), max_index_4[0] + int(area_4.shape[1] * 0.1))]
            regionGrow_4 = regionGrow(area_4, seeds_4, max_4)
            for point in regionGrow_4:
                area_4[point[1], point[0]] = 0
                image[point[1] + int((self.f[1] + self.j[1]) / 2), point[0] + self.b[0]] = 0
                kidneys.append((point[0] + self.b[0], point[1] + int((self.f[1] + self.j[1]) / 2)))

        self.kidneys = kidneys
        self.vertbra = vertbra

    def cut_vertbra(self):
        imageA = np.zeros(self.OriginA.shape, dtype=np.uint8)
        imageP = np.zeros(self.OriginP.shape, dtype=np.uint8)
        for point in self.vertbra:
            imageA[point[1], self.width - 1 - point[0]] = self.OriginA[
                point[1], self.width - 1 - point[0]]
            imageP[point[1], point[0]] = self.OriginP[point[1], point[0]]
        minX = self.width
        maxX = 0
        minY = self.height
        maxY = 0

        for point in self.vertbra_contour:
            if point[0] < minX:
                minX = point[0]
            if point[0] > maxX:
                maxX = point[0]
            if point[1] < minY:
                minY = point[1]
            if point[1] > maxY:
                maxY = point[1]

        vertbraA = imageA[minY:maxY, self.width - 1 - maxX:self.width - 1 - minX]
        vertbraP = imageP[minY:maxY, minX:maxX]

        # 获取脊椎标签
        # labelA = match_2_labels(self.nameA, self.width - 1 - maxX, minY, self.width - 1 - minX, maxY)
        # labelP = match_2_labels(self.nameP, minX, minY, maxX, maxY)
        # print(self.nameP + '--> vertbra A: ' + str(labelA) + ', P: ' + str(labelP))

        return vertbraA, vertbraP

    def cut_pelvis(self):
        pelvA = self.OriginA[self.i[1]:self.m[1], self.o[0]:self.p[0]]
        pelvP = self.OriginP[self.j[1]:self.n[1],
                self.width - self.p[0] - 1:self.width - self.o[0] - 1]

        # 获取骨盆标签
        labelA = match_2_labels(self.nameA, self.o[0], self.i[1], self.p[0], self.m[1])
        labelP = match_2_labels(self.nameP, self.width - self.p[0] - 1, self.j[1],
                                self.width - self.o[0] - 1, self.n[1])

        # mean1, stddv1 = cv.meanStdDev(pelvA)
        # mean2, stddv2 = cv.meanStdDev(pelvP)
        # mean3, stddv3 = cv.meanStdDev(pelvA[int(pelvA.shape[0] * 0.5):int(pelvA.shape[0]),
        #                               int(pelvA.shape[1] * 0.25):int(pelvA.shape[1] * 0.75)])
        # mean4, stddv4 = cv.meanStdDev(pelvP[int(pelvP.shape[0] * 0.5):int(pelvP.shape[0]),
        #                               int(pelvP.shape[1] * 0.25):int(pelvP.shape[1] * 0.75)])
        #
        # if stddv3 - stddv1 > 5:
        #     area_1 = pelvA[int(pelvA.shape[0] * 0.5):int(pelvA.shape[0]),
        #              int(pelvA.shape[1] * 0.25):int(pelvA.shape[1] * 0.75)]
        #     _, max_1, _, max_index_1 = cv.minMaxLoc(area_1)
        #     seeds_1 = [Point(max_index_1[1] + int(pelvA.shape[0] * 0.5),
        #                      max_index_1[0] + int(pelvA.shape[1] * 0.25))]
        #     regionGrow_1 = regionGrow(pelvA, seeds_1, max_1)
        #     for point in regionGrow_1:
        #         pelvA[point[1], point[0]] = 0
        # if stddv4 - stddv2 > 0:
        #     area_2 = pelvP[int(pelvP.shape[0] * 0.5):int(pelvP.shape[0]),
        #              int(pelvP.shape[1] * 0.25):int(pelvP.shape[1] * 0.75)]
        #     _, max_2, _, max_index_2 = cv.minMaxLoc(area_2)
        #     seeds_2 = [Point(max_index_2[1] + int(pelvP.shape[0] * 0.5),
        #                      max_index_2[0] + int(pelvP.shape[1] * 0.25))]
        #     regionGrow_2 = regionGrow(pelvP, seeds_2, max_2)
        #     for point in regionGrow_2:
        #         pelvP[point[1], point[0]] = 0
        #         self.bladder.append((point[0] + self.width - self.p[0] - 1, point[1] + self.j[1]))

        return pelvA, pelvP, [labelA, labelP]

    # 图片边缘修整
    def Cut_Region(self, *imgs):
        images = []
        # 将多余部分剪除
        for image in imgs:
            rows = np.sum(image, axis=1)
            cols = np.sum(image, axis=0)
            minX, minY = 0, 0
            maxY, maxX = image.shape
            # 边界确定
            for digit in rows:
                if digit == 0:
                    minY += 1
                else:
                    break
            while True:
                if maxY == 0:
                    break
                if rows[maxY - 1] == 0:
                    maxY -= 1
                    if maxY == 1:
                        break
                else:
                    break
            for digit in cols:
                if digit == 0:
                    minX += 1
                else:
                    break
            while True:
                if maxX == 0:
                    break
                if cols[maxX - 1] == 0:
                    maxX -= 1
                    if maxX == 1:
                        break
                else:
                    break
            temp = image[minY:maxY, minX:maxX]
            images.append(temp)
        return images

    # 图片Size统一
    def Image_Resize(self, imgs, index):
        ap_row = [168, 93]
        ap_col = [28, 111]

        imgA = cv.resize(imgs[0], (ap_col[index], ap_row[index]), interpolation=cv.INTER_AREA)
        imgP = cv.resize(imgs[1], (ap_col[index], ap_row[index]), interpolation=cv.INTER_AREA)

        # 最后才在原图部位上增强
        imgA = image_enhance(imgA)
        imgP = image_enhance(imgP)

        return [imgA, imgP]

    # 保存部位图像
    def Save_Image(self, path, img_parts):
        cv.imwrite(path + self.nameA, img_parts[0], [int(cv.IMWRITE_JPEG_QUALITY), 95])
        cv.imwrite(path + self.nameP, img_parts[1], [int(cv.IMWRITE_JPEG_QUALITY), 95])

        return
