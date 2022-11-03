from PySide2.QtCore import QFile
from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader


class Stats:
    def __init__(self):
        qfile_stats = QFile("../UI/test1-stats.ui")
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()

        self.ui = QUiLoader().load(qfile_stats)

        self.ui.Button.clicked.connect(self.func1)

    def func1(self):
        info = self.ui.TextEdit.toPlainText()
        print(info)


if __name__ == '__main__':
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()
