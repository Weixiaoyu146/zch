import sys
from bonescantest3 import Ui_mainWindow
from PyQt5 import QtWidgets

class mywindow(QtWidgets.QWidget, Ui_mainWindow):  # 这个地方要注意Ui_MainWindow

    def __init__(self):

        super(mywindow, self).__init__()

        self.setupUi(self)

        #。。。加自己的函数等

if __name__=="__main__":

    app = QtWidgets.QApplication(sys.argv)

    myshow = mywindow()

    myshow.show()#显示

    sys.exit(app.exec_())