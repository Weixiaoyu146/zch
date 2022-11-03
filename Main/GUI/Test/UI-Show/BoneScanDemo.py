import cv2
from PySide2.QtCore import QFile
from PySide2.QtGui import QImage, QPixmap, Qt
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QFrame


class BoneScanDemo:
    def __init__(self):
        q_file = QFile("../UI/BoneScanDemo.ui")
        q_file.open(QFile.ReadOnly)
        q_file.close()

        self.ui = QUiLoader().load(q_file)
        # self.setMinimumSize(1000, 600)

        self.image = cv2.imread("../UI/Images/Mithuha.jpg")

        self.q_image = QImage(self.image, self.image.shape[1], self.image.shape[0], QImage.Format_BGR888)

        self.ui.label.setMinimumSize(540, 540)
        self.ui.label.setAlignment(Qt.AlignCenter)
        self.ui.label.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        self.pixmap = QPixmap(self.q_image).scaled(self.ui.label.size(), aspectMode=Qt.KeepAspectRatio)
        # self.pixmap = QPixmap(self.q_image)
        self.ui.label.setPixmap(self.pixmap)

        self.ui.actionUp.triggered.connect(self.test())
        self.ui.actionDown.triggered.connect(self.test())

    def test(self):
        print(0)


if __name__ == '__main__':
    app = QApplication([])
    demo = BoneScanDemo()
    demo.ui.show()
    app.exec_()
