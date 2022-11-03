from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication


class DicomToJpg:
    def __init__(self):
        ui_file = QFile("../UI/DicomToJpg.ui")
        ui_file.open(QFile.ReadOnly)
        ui_file.close()

        self.ui = QUiLoader().load(ui_file)

        print(0)
        # self.ui.Button.clicked.connect(self.func1)

    def func1(self):
        info = self.ui.TextEdit.toPlainText()
        print(info)


if __name__ == '__main__':
    app = QApplication([])
    convertor = DicomToJpg()
    convertor.ui.show()
    app.exec_()