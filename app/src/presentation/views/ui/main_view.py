# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainui.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QFrame,
    QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

class Ui_deep_fish_widnow(object):
    def setupUi(self, deep_fish_widnow):
        if not deep_fish_widnow.objectName():
            deep_fish_widnow.setObjectName(u"deep_fish_widnow")
        deep_fish_widnow.resize(1006, 736)
        deep_fish_widnow.setStyleSheet(u"QLabel{\n"
"	font-size: 20px;\n"
"}\n"
"\n"
"QCheckBox{\n"
"	font-size: 20px;\n"
"}\n"
"\n"
"QPushButton{\n"
"	font-size: 20px;\n"
"}\n"
"\n"
"QDoubleSpinBox{\n"
"	font-size: 20px;\n"
"	border-radius: 4px;\n"
"	padding: 8px;\n"
"	margin: 8px;\n"
"}\n"
"\n"
"QSpinBox{\n"
"	font-size: 20px;\n"
"	border-radius: 4px;\n"
"	padding: 8px;\n"
"	margin: 8px;\n"
"}\n"
"\n"
"#image_frame{\n"
"	border: 1px solid;\n"
"	border-radius: 8px;\n"
"}")
        self.centralwidget = QWidget(deep_fish_widnow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(10, 10, 10, 10)
        self.control_frame = QFrame(self.centralwidget)
        self.control_frame.setObjectName(u"control_frame")
        self.control_frame.setMaximumSize(QSize(500, 16777215))
        self.control_frame.setFrameShape(QFrame.StyledPanel)
        self.control_frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.control_frame)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.control_inner_frame = QFrame(self.control_frame)
        self.control_inner_frame.setObjectName(u"control_inner_frame")
        self.control_inner_frame.setFrameShape(QFrame.StyledPanel)
        self.control_inner_frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.control_inner_frame)
        self.verticalLayout_2.setSpacing(10)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(10, 0, 10, 0)
        self.smallest_score_frame = QFrame(self.control_inner_frame)
        self.smallest_score_frame.setObjectName(u"smallest_score_frame")
        self.smallest_score_frame.setFrameShape(QFrame.StyledPanel)
        self.smallest_score_frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.smallest_score_frame)
        self.horizontalLayout_3.setSpacing(5)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.smallest_score_label = QLabel(self.smallest_score_frame)
        self.smallest_score_label.setObjectName(u"smallest_score_label")

        self.horizontalLayout_3.addWidget(self.smallest_score_label, 0, Qt.AlignLeft)

        self.smallest_score_spinbox = QDoubleSpinBox(self.smallest_score_frame)
        self.smallest_score_spinbox.setObjectName(u"smallest_score_spinbox")
        self.smallest_score_spinbox.setCursor(QCursor(Qt.ArrowCursor))
        self.smallest_score_spinbox.setMaximum(1.000000000000000)
        self.smallest_score_spinbox.setSingleStep(0.100000000000000)
        self.smallest_score_spinbox.setValue(0.500000000000000)

        self.horizontalLayout_3.addWidget(self.smallest_score_spinbox)


        self.verticalLayout_2.addWidget(self.smallest_score_frame)

        self.bounding_box_number_frame = QFrame(self.control_inner_frame)
        self.bounding_box_number_frame.setObjectName(u"bounding_box_number_frame")
        self.bounding_box_number_frame.setFrameShape(QFrame.StyledPanel)
        self.bounding_box_number_frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_4 = QHBoxLayout(self.bounding_box_number_frame)
        self.horizontalLayout_4.setSpacing(5)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.bounding_box_number_label = QLabel(self.bounding_box_number_frame)
        self.bounding_box_number_label.setObjectName(u"bounding_box_number_label")

        self.horizontalLayout_4.addWidget(self.bounding_box_number_label, 0, Qt.AlignLeft)

        self.bounding_box_number_spinbox = QSpinBox(self.bounding_box_number_frame)
        self.bounding_box_number_spinbox.setObjectName(u"bounding_box_number_spinbox")
        self.bounding_box_number_spinbox.setMaximum(10)

        self.horizontalLayout_4.addWidget(self.bounding_box_number_spinbox)


        self.verticalLayout_2.addWidget(self.bounding_box_number_frame)

        self.display_all_frame = QFrame(self.control_inner_frame)
        self.display_all_frame.setObjectName(u"display_all_frame")
        self.display_all_frame.setFrameShape(QFrame.StyledPanel)
        self.display_all_frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_5 = QHBoxLayout(self.display_all_frame)
        self.horizontalLayout_5.setSpacing(5)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.display_all_check_box = QCheckBox(self.display_all_frame)
        self.display_all_check_box.setObjectName(u"display_all_check_box")

        self.horizontalLayout_5.addWidget(self.display_all_check_box, 0, Qt.AlignHCenter)


        self.verticalLayout_2.addWidget(self.display_all_frame)


        self.verticalLayout.addWidget(self.control_inner_frame)

        self.select_image_button = QPushButton(self.control_frame)
        self.select_image_button.setObjectName(u"select_image_button")
        self.select_image_button.setMaximumSize(QSize(16777215, 16777215))
        self.select_image_button.setCursor(QCursor(Qt.PointingHandCursor))

        self.verticalLayout.addWidget(self.select_image_button)


        self.horizontalLayout.addWidget(self.control_frame)

        self.image_frame = QFrame(self.centralwidget)
        self.image_frame.setObjectName(u"image_frame")
        self.image_frame.setMinimumSize(QSize(0, 0))
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.image_frame)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel(self.image_frame)
        self.image_label.setObjectName(u"image_label")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_label.sizePolicy().hasHeightForWidth())
        self.image_label.setSizePolicy(sizePolicy)
        self.image_label.setMinimumSize(QSize(510, 0))
        self.image_label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_2.addWidget(self.image_label)


        self.horizontalLayout.addWidget(self.image_frame)

        deep_fish_widnow.setCentralWidget(self.centralwidget)

        self.retranslateUi(deep_fish_widnow)

        QMetaObject.connectSlotsByName(deep_fish_widnow)
    # setupUi

    def retranslateUi(self, deep_fish_widnow):
        deep_fish_widnow.setWindowTitle(QCoreApplication.translate("deep_fish_widnow", u"MainWindow", None))
        self.smallest_score_label.setText(QCoreApplication.translate("deep_fish_widnow", u"Smallest Score", None))
        self.bounding_box_number_label.setText(QCoreApplication.translate("deep_fish_widnow", u"Bounding Box Numbers", None))
        self.display_all_check_box.setText(QCoreApplication.translate("deep_fish_widnow", u"Display All", None))
        self.select_image_button.setText(QCoreApplication.translate("deep_fish_widnow", u"Select Image", None))
        self.image_label.setText(QCoreApplication.translate("deep_fish_widnow", u"No Image", None))
    # retranslateUi

