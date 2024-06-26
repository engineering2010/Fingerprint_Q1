# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Recognition.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Recognition(object):
    def setupUi(self, Recognition):
        Recognition.setObjectName("Recognition")
        Recognition.resize(970, 868)
        self.groupBox_2 = QtWidgets.QGroupBox(Recognition)
        self.groupBox_2.setGeometry(QtCore.QRect(40, 160, 331, 221))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_Evaluate = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_Evaluate.setGeometry(QtCore.QRect(60, 130, 161, 31))
        self.pushButton_Evaluate.setObjectName("pushButton_Evaluate")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(40, 80, 81, 31))
        self.label_3.setObjectName("label_3")
        self.textEdit_Threshold = QtWidgets.QTextEdit(self.groupBox_2)
        self.textEdit_Threshold.setGeometry(QtCore.QRect(140, 80, 141, 31))
        self.textEdit_Threshold.setObjectName("textEdit_Threshold")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(40, 40, 91, 31))
        self.label.setObjectName("label")
        self.textEdit_Name = QtWidgets.QTextEdit(self.groupBox_2)
        self.textEdit_Name.setGeometry(QtCore.QRect(140, 40, 141, 31))
        self.textEdit_Name.setObjectName("textEdit_Name")
        self.pushButton_Draw = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_Draw.setGeometry(QtCore.QRect(60, 170, 161, 41))
        self.pushButton_Draw.setObjectName("pushButton_Draw")
        self.groupBox_3 = QtWidgets.QGroupBox(Recognition)
        self.groupBox_3.setGeometry(QtCore.QRect(420, 10, 491, 371))
        self.groupBox_3.setObjectName("groupBox_3")
        self.textEdit_Results = QtWidgets.QTextEdit(self.groupBox_3)
        self.textEdit_Results.setGeometry(QtCore.QRect(10, 60, 471, 301))
        self.textEdit_Results.setObjectName("textEdit_Results")
        self.pushButton_Clear = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_Clear.setGeometry(QtCore.QRect(210, 30, 75, 24))
        self.pushButton_Clear.setObjectName("pushButton_Clear")
        self.groupBox = QtWidgets.QGroupBox(Recognition)
        self.groupBox.setGeometry(QtCore.QRect(40, 10, 331, 141))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_Enroll = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_Enroll.setGeometry(QtCore.QRect(60, 40, 171, 31))
        self.pushButton_Enroll.setObjectName("pushButton_Enroll")
        self.pushButton__Match = QtWidgets.QPushButton(self.groupBox)
        self.pushButton__Match.setGeometry(QtCore.QRect(60, 90, 171, 31))
        self.pushButton__Match.setObjectName("pushButton__Match")
        self.groupBox_4 = QtWidgets.QGroupBox(Recognition)
        self.groupBox_4.setGeometry(QtCore.QRect(40, 390, 881, 411))
        self.groupBox_4.setObjectName("groupBox_4")

        self.retranslateUi(Recognition)
        QtCore.QMetaObject.connectSlotsByName(Recognition)

    def retranslateUi(self, Recognition):
        _translate = QtCore.QCoreApplication.translate
        Recognition.setWindowTitle(_translate("Recognition", "Form"))
        self.groupBox_2.setTitle(_translate("Recognition", "Evaluate fingerprint recognition system"))
        self.pushButton_Evaluate.setText(_translate("Recognition", "Start to match"))
        self.label_3.setText(_translate("Recognition", "Threshold:"))
        self.label.setText(_translate("Recognition", "Match Name:"))
        self.pushButton_Draw.setText(_translate("Recognition", "Draw a ROC curve"))
        self.groupBox_3.setTitle(_translate("Recognition", "Matched fingerprints"))
        self.pushButton_Clear.setText(_translate("Recognition", "Clear"))
        self.groupBox.setTitle(_translate("Recognition", "Enrol and match fingerprint"))
        self.pushButton_Enroll.setText(_translate("Recognition", "Enrol a fingerprint"))
        self.pushButton__Match.setText(_translate("Recognition", "Compare fingerpirnts"))
        self.groupBox_4.setTitle(_translate("Recognition", "ROC curve"))
