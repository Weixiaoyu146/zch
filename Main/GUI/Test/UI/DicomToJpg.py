# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'DicomToJpg.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################
import os
import shutil

import cv2
import numpy as np
import pydicom
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class DicomToJpg(QWidget):
    def __init__(self, parent=None):
        super(DicomToJpg, self).__init__(parent)
        self.setWindowTitle("DicomToJpg转换器")
        self.resize(600, 41)
        icon = QIcon("Images/UESTC2.jpg")
        self.setWindowIcon(icon)

        self.label = QLabel()
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(20)
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText("请先选择保存jpg图像的文件夹")

        self.pushButton = QPushButton()
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.pushButton.setSizePolicy(sizePolicy1)
        self.pushButton.setText("选择文件夹")
        self.pushButton.clicked.connect(self.open_dir)

        self.pushButton_2 = QPushButton()
        self.pushButton_2.setSizePolicy(sizePolicy1)
        self.pushButton_2.setText("转换")
        self.pushButton_2.clicked.connect(self.convert)

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.pushButton)
        layout.addWidget(self.pushButton_2)
        self.setLayout(layout)

        self.save_dir = ""

    def open_dir(self):
        self.save_dir = QFileDialog.getExistingDirectory(self,
                                                         "选择保存jpg图像的文件夹",
                                                         os.getcwd(),
                                                         QFileDialog.ShowDirsOnly)
        self.label.setText(self.save_dir)

    def convert(self):
        if self.save_dir == "":
            QMessageBox.warning(self, "提醒", "请先选择保存jpg图像的文件夹", QMessageBox.Ok, QMessageBox.Ok)
        else:
            dicom_dir = QFileDialog.getExistingDirectory(self,
                                                         "选择含Dicom图像的文件夹",
                                                         os.getcwd(),
                                                         QFileDialog.ShowDirsOnly)
            if dicom_dir != "":
                image_save_dir = os.path.join(self.save_dir, r"Images")
                dicom_save_dir = os.path.join(self.save_dir, r"Dicoms")
                if not os.path.exists(image_save_dir):
                    os.makedirs(image_save_dir)
                if not os.path.exists(dicom_save_dir):
                    os.makedirs(dicom_save_dir)
                self.search_dicom(dicom_dir, dicom_save_dir)
                self.dicom_to_jpg(dicom_save_dir, image_save_dir)
                QMessageBox.information(self, "消息", "转换完成", QMessageBox.Ok, QMessageBox.Ok)

    def search_dicom(self, dicom_dir, dicom_save_dir):
        for dir_or_file in os.listdir(dicom_dir):
            dir_or_file_path = os.path.join(dicom_dir, dir_or_file)
            if os.path.isfile(dir_or_file_path):
                if os.path.basename(dir_or_file_path).endswith('.DCM'):
                    shutil.copyfile(dir_or_file_path, os.path.join(dicom_save_dir, os.path.basename(dir_or_file_path)))
                else:
                    continue
            elif os.path.isdir(dir_or_file_path):
                self.search_dicom(dir_or_file_path, dicom_save_dir)

    def dicom_to_jpg(self, dicom_save_dir, image_save_dir):
        for dicom in os.listdir(dicom_save_dir):
            dicom_path = os.path.join(dicom_save_dir, dicom)
            dicom_info = pydicom.dcmread(dicom_path)
            try:
                anterior_binary_image, posterior_binary_image = dicom_info.pixel_array
            except:
                print(dicom_path)
                continue
            patient_id = dicom_info.PatientID
            patient_name = dicom.split('.')[-2].split('-')[-1]
            anterior_image = np.uint8(anterior_binary_image)
            posterior_image = np.uint8(posterior_binary_image)
            images = np.hstack((anterior_image, posterior_image))
            white_images = ~images
            white_images_path = os.path.join(image_save_dir, patient_id + r'-' + patient_name + r".jpg")
            cv2.imwrite(white_images_path, white_images, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    app = QApplication([])
    demo = DicomToJpg()
    demo.show()
    app.exec_()
