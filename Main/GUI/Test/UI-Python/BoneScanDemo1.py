# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'BoneScanDemo1.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1530, 900)
        icon = QIcon()
        icon.addFile(u"../UI/Images/UESTC2.jpg", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        self.actionOpenDir = QAction(MainWindow)
        self.actionOpenDir.setObjectName(u"actionOpenDir")
        self.actionLoadModel = QAction(MainWindow)
        self.actionLoadModel.setObjectName(u"actionLoadModel")
        self.actionReport = QAction(MainWindow)
        self.actionReport.setObjectName(u"actionReport")
        self.actionPrint = QAction(MainWindow)
        self.actionPrint.setObjectName(u"actionPrint")
        self.actionPredict = QAction(MainWindow)
        self.actionPredict.setObjectName(u"actionPredict")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label)

        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.scrollArea = QScrollArea(self.groupBox)
        self.scrollArea.setObjectName(u"scrollArea")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollArea.sizePolicy().hasHeightForWidth())
        self.scrollArea.setSizePolicy(sizePolicy)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 162, 820))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.listView = QListView(self.scrollAreaWidgetContents)
        self.listView.setObjectName(u"listView")

        self.verticalLayout.addWidget(self.listView)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.horizontalLayout_2.addWidget(self.scrollArea)

        self.graphicsView_source = QGraphicsView(self.groupBox)
        self.graphicsView_source.setObjectName(u"graphicsView_source")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(4)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.graphicsView_source.sizePolicy().hasHeightForWidth())
        self.graphicsView_source.setSizePolicy(sizePolicy1)

        self.horizontalLayout_2.addWidget(self.graphicsView_source)

        self.graphicsView_target = QGraphicsView(self.groupBox)
        self.graphicsView_target.setObjectName(u"graphicsView_target")
        sizePolicy1.setHeightForWidth(self.graphicsView_target.sizePolicy().hasHeightForWidth())
        self.graphicsView_target.setSizePolicy(sizePolicy1)

        self.horizontalLayout_2.addWidget(self.graphicsView_target)


        self.verticalLayout_2.addWidget(self.groupBox)

        MainWindow.setCentralWidget(self.centralwidget)
        self.toolBar = QToolBar(MainWindow)
        self.toolBar.setObjectName(u"toolBar")
        self.toolBar.setMovable(False)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolBar)

        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionOpenDir)
        self.toolBar.addAction(self.actionLoadModel)
        self.toolBar.addAction(self.actionPredict)
        self.toolBar.addAction(self.actionReport)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u6838\u7d20\u9aa8\u626b\u63cf\u56fe\u50cf\u70ed\u70b9\u68c0\u6d4b\u7cfb\u7edf - \u7535\u5b50\u79d1\u6280\u5927\u5b66\u5065\u5eb7\u5927\u6570\u636e\u7814\u7a76\u6240", None))
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"Open", None))
#if QT_CONFIG(tooltip)
        self.actionOpen.setToolTip(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u6587\u4ef6", None))
#endif // QT_CONFIG(tooltip)
        self.actionOpenDir.setText(QCoreApplication.translate("MainWindow", u"Open Directory", None))
#if QT_CONFIG(tooltip)
        self.actionOpenDir.setToolTip(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u6587\u4ef6\u5939", None))
#endif // QT_CONFIG(tooltip)
        self.actionLoadModel.setText(QCoreApplication.translate("MainWindow", u"Load Model", None))
#if QT_CONFIG(tooltip)
        self.actionLoadModel.setToolTip(QCoreApplication.translate("MainWindow", u"\u9009\u62e9\u6a21\u578b", None))
#endif // QT_CONFIG(tooltip)
        self.actionReport.setText(QCoreApplication.translate("MainWindow", u"Report", None))
#if QT_CONFIG(tooltip)
        self.actionReport.setToolTip(QCoreApplication.translate("MainWindow", u"\u751f\u6210\u62a5\u544a", None))
#endif // QT_CONFIG(tooltip)
        self.actionPrint.setText(QCoreApplication.translate("MainWindow", u"Print", None))
#if QT_CONFIG(tooltip)
        self.actionPrint.setToolTip(QCoreApplication.translate("MainWindow", u"\u6253\u5370\u62a5\u544a", None))
#endif // QT_CONFIG(tooltip)
        self.actionPredict.setText(QCoreApplication.translate("MainWindow", u"Predict", None))
#if QT_CONFIG(tooltip)
        self.actionPredict.setToolTip(QCoreApplication.translate("MainWindow", u"\u68c0\u6d4b\u70ed\u70b9", None))
#endif // QT_CONFIG(tooltip)
        self.label.setText(QCoreApplication.translate("MainWindow", u"Directory", None))
        self.groupBox.setTitle("")
        self.toolBar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
    # retranslateUi

