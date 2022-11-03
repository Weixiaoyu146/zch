import os
import cv2

from PySide2.QtGui import QPixmap, QImage, QPicture
from PySide2.QtWidgets import QApplication, QAction, QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, Qt, QEvent


class BoneScanDemo:
    def __init__(self):
        self.cwd = os.getcwd()
        self.image = []
        self.image_path = ""
        q_file = QFile("../UI/BoneScanDemo1.ui")
        q_file.open(QFile.ReadOnly)
        q_file.close()
        self.ui = QUiLoader().load(q_file)
        self.setup_ui()

    def setup_ui(self):
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionOpenDir.triggered.connect(self.open_file)
        self.ui.actionReport.triggered.connect(self.open_file)
        self.ui.actionPrint.triggered.connect(self.open_file)

    def open_file(self):
        image_paths = QFileDialog.getOpenFileName(self.ui,
                                                  "选择文件",
                                                  self.cwd,
                                                  "JPEG Files(*.jpg);;Dicom Files(*.dcm)")
        if image_paths[1] == "JPEG Files(*.jpg)":
            self.image_path = image_paths[0]
            self.image = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
            self.show_image()
            # print(self.image.shape)

    def show_image(self):
        q_image = QImage(self.image, self.image.shape[1], self.image.shape[0], QImage.Format_Grayscale8)
        q_pixmap = QPixmap(q_image)
        self.ui.label_source_image.setPixmap(q_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            print(0)


if __name__ == '__main__':
    app = QApplication([])
    demo = BoneScanDemo()
    demo.ui.show()
    app.exec_()
