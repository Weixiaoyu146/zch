# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bonescantest1.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import Image
import pydicom
import numpy as np
import cv2
import scipy.misc
import os
import matplotlib.pyplot as plt


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("UESTC2.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 20, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(90, 20, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 530, 54, 12))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(280, 530, 54, 12))
        self.label_2.setObjectName("label_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(170, 20, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(420, 530, 75, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(515, 530, 75, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(710, 530, 75, 23))
        self.pushButton_6.setObjectName("pushButton_6")
        # self.treeView = QtWidgets.QTreeView(self.centralwidget)
        # self.treeView.setGeometry(QtCore.QRect(10, 50, 191, 471))
        # self.treeView.setObjectName("treeView")

        self.label_3 = QLabel(self)
        self.label_3.setFixedSize(191, 471)
        self.label_3.move(10, 50)

        self.label_4 = QLabel(self)
        self.label_4.setFixedSize(191, 471)
        self.label_4.move(210, 50)

        # self.treeView_2 = QtWidgets.QTreeView(self.centralwidget)
        # self.treeView_2.setGeometry(QtCore.QRect(210, 50, 191, 471))
        # self.treeView_2.setObjectName("treeView_2")

        self.treeView_3 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_3.setGeometry(QtCore.QRect(470, 50, 71, 41))
        self.treeView_3.setObjectName("treeView_3")
        self.treeView_4 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_4.setGeometry(QtCore.QRect(420, 100, 61, 31))
        self.treeView_4.setObjectName("treeView_4")
        self.treeView_5 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_5.setGeometry(QtCore.QRect(420, 140, 61, 91))
        self.treeView_5.setObjectName("treeView_5")
        self.treeView_6 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_6.setGeometry(QtCore.QRect(490, 100, 31, 171))
        self.treeView_6.setObjectName("treeView_6")
        self.treeView_7 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_7.setGeometry(QtCore.QRect(530, 100, 61, 31))
        self.treeView_7.setObjectName("treeView_7")
        self.treeView_8 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_8.setGeometry(QtCore.QRect(530, 140, 61, 91))
        self.treeView_8.setObjectName("treeView_8")
        self.treeView_9 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_9.setGeometry(QtCore.QRect(420, 240, 61, 31))
        self.treeView_9.setObjectName("treeView_9")
        self.treeView_10 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_10.setGeometry(QtCore.QRect(530, 240, 61, 31))
        self.treeView_10.setObjectName("treeView_10")
        self.treeView_11 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_11.setGeometry(QtCore.QRect(420, 360, 61, 31))
        self.treeView_11.setObjectName("treeView_11")
        self.treeView_12 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_12.setGeometry(QtCore.QRect(530, 360, 61, 31))
        self.treeView_12.setObjectName("treeView_12")
        self.treeView_13 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_13.setGeometry(QtCore.QRect(420, 490, 61, 31))
        self.treeView_13.setObjectName("treeView_13")
        self.treeView_14 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_14.setGeometry(QtCore.QRect(530, 490, 61, 31))
        self.treeView_14.setObjectName("treeView_14")
        self.treeView_15 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_15.setGeometry(QtCore.QRect(450, 280, 111, 61))
        self.treeView_15.setObjectName("treeView_15")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(610, 50, 181, 471))
        self.textEdit.setObjectName("textEdit")
        #MainWindow.setCentralWidget(self.centralwidget)
        #self.menubar = QtWidgets.QMenuBar(MainWindow)
        #self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        #self.menubar.setObjectName("menubar")
        #MainWindow.setMenuBar(self.menubar)
        #self.statusbar = QtWidgets.QStatusBar(MainWindow)
        #self.statusbar.setObjectName("statusbar")
        #MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "骨扫描识别检测系统"))
        self.pushButton.setText(_translate("MainWindow", "打开图像"))
        self.pushButton_2.setText(_translate("MainWindow", "勾画热点"))
        self.label.setText(_translate("MainWindow", "前身图像"))
        self.label_2.setText(_translate("MainWindow", "后身图像"))
        self.pushButton_3.setText(_translate("MainWindow", "生成报告"))
        self.pushButton_4.setText(_translate("MainWindow", "前身分割"))
        self.pushButton_5.setText(_translate("MainWindow", "后身分割"))
        self.pushButton_6.setText(_translate("MainWindow", "提交"))

    def openfile(self):
        openfile_name = QFileDialog.getOpenFileName(self,'选择文件','','dicom files(*.dcm)')

        # openfile_name = QFileDialog.getOpenFileName(self,'选择文件','','jpg files(*.jpg)')
        # jpg = QtGui.QPixmap(openfile_name[0]).scaled(self.label_3.width(), self.label_3.height())
        # self.label_3.setPixmap(jpg)

        dicom_image = pydicom.dcmread(openfile_name[0])
        dicom_double_image = dicom_image.pixel_array
        dicom_double_image.flags.writeable = True
        AImg = dicom_double_image[0]
        PImg = dicom_double_image[1]

        A_img_path = os.path.splitext(openfile_name[0])[0] + 'Aimg.jpg'
        P_img_path = os.path.splitext(openfile_name[0])[0] + 'Pimg.jpg'
        # plt.imshow(dicom_double_image)
        # plt.show()
        # scipy.misc.imsave(A_img_path, AImg)
        # scipy.misc.imsave(P_img_path, PImg)

        # Anterior_image = Image.fromarray(AImg.astype('uint8')).convert('RGB')
        # Posterior_image = Image.fromarray(PImg.astype('uint8')).convert('RGB')

        Anterior_image = np.uint8(AImg)
        Posterior_image = np.uint8(PImg)
        cv2.imencode('.jpg', Anterior_image)[1].tofile(A_img_path)
        cv2.imencode('.jpg', Posterior_image)[1].tofile(P_img_path)
        # Anterior_img = Image.open(A_img_path)
        # Posterior_img = Image.open(P_img_path)
        # Anterior_image = cv2.imdecode(Anterior_img, cv2.IMREAD_COLOR)
        # Posterior_image = cv2.imdecode(Posterior_img, cv2.IMREAD_COLOR)
        Anterior_image_jpg = QtGui.QPixmap(A_img_path).scaled(self.label_3.width(), self.label_3.height())
        Posterior_image_jpg = QtGui.QPixmap(P_img_path).scaled(self.label_4.width(), self.label_4.height())
        self.label_3.setPixmap(Anterior_image_jpg)  # Label设置Pixmap，显示图片
        self.label_4.setPixmap(Posterior_image_jpg)  # Label设置Pixmap，显示图片6

        # print(openfile_name[0])

