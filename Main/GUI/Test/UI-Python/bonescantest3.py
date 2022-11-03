# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bonescantest3.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from ImageProcessing import *
import os


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1200, 900)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("UESTC2.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(220, 11, 751, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(60, 770, 1081, 91))
        self.textEdit.setObjectName("textEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1060, 870, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(60, 10, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.get_original_image)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(970, 10, 41, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.get_image_saving_path)
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(60, 40, 512, 712))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setMinimumSize(QtCore.QSize(210, 710))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setMinimumSize(QtCore.QSize(210, 710))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(630, 40, 512, 712))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_3.setMinimumSize(QtCore.QSize(210, 710))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_4.setMinimumSize(QtCore.QSize(210, 710))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(160, 11, 54, 21))
        self.label_5.setObjectName("label_5")

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "骨扫描识别与检测系统"))
        self.pushButton.setText(_translate("mainWindow", "提交"))
        self.pushButton_2.setText(_translate("mainWindow", "打开图像"))
        self.pushButton_3.setText(_translate("mainWindow", "浏览"))
        self.label.setStyleSheet('border-width: 1px;border-style: solid; border-color: rgb(255, 170, 0);')
        self.label_2.setStyleSheet('border-width: 1px;border-style: solid; border-color: rgb(255, 170, 0);')
        self.label_3.setStyleSheet('border-width: 1px;border-style: solid; border-color: rgb(255, 170, 0);')
        self.label_4.setStyleSheet('border-width: 1px;border-style: solid; border-color: rgb(255, 170, 0);')
        self.label_5.setText(_translate("mainWindow", "图片路径："))

    def get_image_saving_path(self):

        image_saving_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "D:\Pictures")
        self.lineEdit.setText(image_saving_path)

    def get_original_image(self):
        image_saving_path = self.lineEdit.text()
        if os.path.exists(image_saving_path):
            dicom_image_path = QFileDialog.getOpenFileName(self,'选择文件','','dicom files(*.dcm)')
        else:
            QMessageBox.warning(self,
                                    "提示",
                                    "请先设置图片保存路径！",
                                    QMessageBox.Yes)
            return
        if dicom_image_path[0] != '':
            image_dictory = os.path.splitext(dicom_image_path[0])[0]
            original_anterior_image, original_posterior_image = dicom_to_opencv_gray(dicom_image_path[0], image_dictory)
            preprocessed_anterior_image, preprocessed_posterior_image, preprocessed_anterior_image_mask, preprocessed_posterior_image_mask = cut_preprocessing_image(original_anterior_image, original_posterior_image, image_dictory)

            segmentataion = Segmentation(preprocessed_anterior_image_mask, preprocessed_posterior_image_mask, preprocessed_anterior_image, preprocessed_posterior_image, image_dictory)
            predict = segmentataion.Segment()
            text = ''
            ap_or_pa = ['前身', '后身']
            part_image_name = ['头部', '左肩', '右肩', '左胸', '右胸', '左肘', '右肘',
                               '脊柱', '骨盆', '左膝', '右膝', '左踝', '右踝']
            # for i in range(26):
            #     if i >= 13:
            #         if predict[i][0] >= 0.75:
            #             text += ap_or_pa[0] + part_image_name[i] + '正常;    '
            #         else:
            #             text += ap_or_pa[0] + part_image_name[i] + '疑似热点;'
            #     else:
            #         if predict[i][0] >= 0.75:
            #             text += ap_or_pa[1] + part_image_name[i] + '正常;    '
            #         else:
            #             text += ap_or_pa[1] + part_image_name[i] + '疑似热点;'

            # if predict[0][0] >= 0.75:
            #     text += '前身头部正常;'
            # else:
            #     text += '前身头部疑似热点;'
            anterior_image_path = image_dictory + r'/Image/Preprocessed/anterior_image.jpg'
            posterior_image_path = image_dictory + r'/Image/Preprocessed/posterior_image.jpg'
            segmentataion_anterior_image_path = image_dictory + r'/Image/Segmentation/anterior_image.jpg'
            segmentataion_posterior_image_path = image_dictory + r'/Image/Segmentation/posterior_image.jpg'

            self.label.setFixedSize(preprocessed_anterior_image.shape[1], preprocessed_anterior_image.shape[0])
            self.label_2.setFixedSize(preprocessed_anterior_image.shape[1], preprocessed_anterior_image.shape[0])
            self.label_3.setFixedSize(preprocessed_anterior_image.shape[1], preprocessed_anterior_image.shape[0])
            self.label_4.setFixedSize(preprocessed_anterior_image.shape[1], preprocessed_anterior_image.shape[0])

            anterior_image_label = QtGui.QPixmap(anterior_image_path).scaled(self.label.width(), self.label.height())
            posterior_image_label = QtGui.QPixmap(posterior_image_path).scaled(self.label_2.width(), self.label_2.height())
            segmentataion_anterior_image_label = QtGui.QPixmap(segmentataion_anterior_image_path).scaled(self.label_3.width(), self.label_3.height())
            segmentataion_posterior_image_label = QtGui.QPixmap(segmentataion_posterior_image_path).scaled(self.label_4.width(), self.label_4.height())
            self.label.setPixmap(anterior_image_label)  # Label设置Pixmap，显示图片
            self.label_2.setPixmap(posterior_image_label)  # Label设置Pixmap，显示图片
            self.label_3.setPixmap(segmentataion_anterior_image_label)  # Label设置Pixmap，显示图片
            self.label_4.setPixmap(segmentataion_posterior_image_label)  # Label设置Pixmap，显示图片
            # self.textEdit.setText(text)
        return