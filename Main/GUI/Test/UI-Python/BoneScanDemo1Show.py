import os

import numpy as np
import pydicom
import cv2
from PySide2 import QtWidgets
from PySide2.QtCore import QDir
from PySide2.QtGui import QImage, QPixmap, Qt, QCursor
from PySide2.QtWidgets import QFileDialog, QFrame, QGraphicsPixmapItem, QGraphicsScene, QApplication, QGraphicsView, \
    QFileSystemModel

from BoneScanDemo1 import Ui_MainWindow


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):  # 这个地方要注意Ui_MainWindow

    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.reSetupUi()

        self.cwd = os.getcwd()

    def reSetupUi(self):
        self.actionOpen.triggered.connect(self.open_file)
        self.actionOpenDir.triggered.connect(self.open_dir)
        self.actionLoadModel.triggered.connect(self.load_model)
        self.actionPredict.triggered.connect(self.predict)
        # self.actionReport.triggered.connect(self.generate_report)
        # self.actionPrint.triggered.connect(self.print_report)

    def open_file(self):
        file_paths = QFileDialog.getOpenFileName(self,
                                                 "Open file",
                                                 self.cwd,
                                                 "Images(*.jpg *.dcm)")
        if file_paths[0][-4:] == ".jpg":
            image_path = file_paths[0]
            self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        elif file_paths[0][-4:] == ".dcm":
            dicom_path = file_paths[0]
            if "ANTERIOR" in dicom_path:
                for file in os.listdir(dicom_path[:-19]):
                    if "POST" in file:
                        dicom_path_p = dicom_path[:-18] + file
                        break
                anterior_binary_image = pydicom.dcmread(dicom_path).pixel_array
                posterior_binary_image = pydicom.dcmread(dicom_path_p).pixel_array
            # TODO POST要加逻辑
            else:
                anterior_binary_image, posterior_binary_image = pydicom.dcmread(dicom_path).pixel_array

            anterior_image = np.uint8(anterior_binary_image)
            posterior_image = np.uint8(posterior_binary_image)
            if anterior_image.shape[1] == 512:
                anterior_image = anterior_image[:, 128:384]
                posterior_image = posterior_image[:, 127:383]
            self.image = np.hstack((anterior_image, posterior_image))
        if self.image is not None:
            self.show_image()

    def open_dir(self):
        files_dir = QFileDialog.getExistingDirectory(self,
                                                      "Open directory",
                                                      self.cwd,
                                                      QFileDialog.ShowDirsOnly)
        self.label.setText(files_dir)
        self.dir_model = QFileSystemModel()
        self.dir_model.setRootPath(files_dir)
        self.dir_model.setFilter(QDir.NoDotAndDotDot | QDir.Files)
        self.dir_model.setNameFilters(['*.jpg', '*.dcm'])
        self.dir_model.setNameFilterDisables(False)

        self.listView.setModel(self.dir_model)
        self.listView.setRootIndex(self.dir_model.index(files_dir))
        self.listView.doubleClicked.connect(self.select_file)

    def load_model(self):
        model_paths = QFileDialog.getOpenFileName(self,
                                                  "Load model",
                                                  self.cwd,
                                                  "Models(*.pth *.pkl)")
        print(model_paths)

    def predict(self):
        file_paths = QFileDialog.getOpenFileName(self,
                                                 "Open file",
                                                 self.cwd,
                                                 "Images(*.jpg)")
        if file_paths[1] == "Images(*.jpg)":
            image_path_final = file_paths[0]
            self.image_final = cv2.imread(image_path_final, cv2.IMREAD_COLOR)
            self.show_image_final()

    def generate_report(self):
        print("report")

    def print_report(self):
        print("print")

    def select_file(self, Qmodelidx):
        # image_path = self.dir_model.filePath(Qmodelidx)
        # self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # self.show_image()
        file_path = self.dir_model.filePath(Qmodelidx)
        if file_path[-4:] == ".jpg":
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        elif file_path[-4:] == ".dcm":
            if "ANTERIOR" in file_path:
                for file in os.listdir(file_path[:-19]):
                    if "POST" in file:
                        file_path_post = file_path[:-18] + file
                        break
                anterior_binary_image = pydicom.dcmread(file_path).pixel_array
                posterior_binary_image = pydicom.dcmread(file_path_post).pixel_array
            # TODO POST要加逻辑
            else:
                anterior_binary_image, posterior_binary_image = pydicom.dcmread(file_path).pixel_array

            anterior_image = np.uint8(anterior_binary_image)
            posterior_image = np.uint8(posterior_binary_image)
            if anterior_image.shape[1] == 512:
                anterior_image = anterior_image[:, 128:384]
                posterior_image = posterior_image[:, 127:383]
            self.image = np.hstack((anterior_image, posterior_image))
        if self.image is not None:
            self.show_image()

    def show_image(self):
        q_image = QImage(self.image, self.image.shape[1], self.image.shape[0], QImage.Format_Grayscale8)
        q_pixmap = QPixmap(q_image)
        scene = QGraphicsScene()
        scene.addPixmap(q_pixmap)
        self.graphicsView_source.setScene(scene)

    def show_image_final(self):
        q_image = QImage(self.image_final, self.image_final.shape[1], self.image_final.shape[0],
                         3 * self.image_final.shape[1], QImage.Format_BGR888).scaledToHeight(1024)
        q_pixmap = QPixmap(q_image)
        scene = QGraphicsScene()
        scene.addPixmap(q_pixmap)
        self.graphicsView_target.setScene(scene)

    # def zoom_in(self):
    #     print("zoom in")
    #
    # def zoom_out(self):
    #     print("zoom out")
    #
    # def mousePressEvent(self, event):
    #     if event.button() == Qt.LeftButton:
    #         # self.graphicsView_source.centerOn()
    #         print(self.graphicsView_source.scene())
    #
    # def wheelEvent(self, event):
    #     # print(self.graphicsView_source.sceneRect())
    #     # print(self.graphicsView_source.geometry())
    #     # print(self.graphicsView_source.mapToScene(event.pos()))
    #     # print(event.pos())
    #
    #     # print(self.mapFromGlobal(QCursor.pos()))
    #     # print(self.toolBar.size())
    #     # print(self.graphicsView_source.itemAt(event.p))
    #     # print(self.mapFrom(self.centralwidget, event.pos()))
    #     # print(self.mapFrom(self.graphicsView_source, event.pos()))
    #     # if self.graphicsView_source.geometry().contains(self.mapToGlobal(event.pos())):
    #     #     print(0)
    #     if QApplication.keyboardModifiers() == Qt.ControlModifier:
    #         # print(event.delta())
    #         if event.delta() > 0:
    #             self.zoom_in()
    #         else:
    #             self.zoom_out()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    demo = mywindow()
    demo.show()  # 显示
    app.exec_()
