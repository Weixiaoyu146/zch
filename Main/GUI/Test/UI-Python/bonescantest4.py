# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bonescantest4.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1200, 900)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(220, 11, 751, 21))
        self.lineEdit.setObjectName("lineEdit")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(60, 770, 1081, 91))
        self.textEdit.setObjectName("textEdit")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(160, 11, 54, 21))
        self.label_5.setObjectName("label_5")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1060, 870, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(60, 40, 512, 712))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setMinimumSize(QtCore.QSize(210, 710))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setMinimumSize(QtCore.QSize(210, 710))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
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
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(60, 10, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(970, 10, 41, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        mainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "骨扫描识别与检测系统"))
        self.label.setText(_translate("mainWindow", "TextLabel"))
        self.label_2.setText(_translate("mainWindow", "TextLabel"))
        self.label_3.setText(_translate("mainWindow", "TextLabel"))
        self.label_4.setText(_translate("mainWindow", "TextLabel"))
        self.label_5.setText(_translate("mainWindow", "图片路径："))
        self.pushButton.setText(_translate("mainWindow", "提交"))
        self.pushButton_2.setText(_translate("mainWindow", "打开图像"))
        self.pushButton_3.setText(_translate("mainWindow", "浏览"))
