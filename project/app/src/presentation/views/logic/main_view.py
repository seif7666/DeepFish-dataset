import os
from typing import List
from PySide6.QtWidgets import QMainWindow, QFileDialog
from PySide6.QtGui import QPixmap
from app.src.presentation.views.ui.main_view import Ui_deep_fish_window as View
from app.src.core.helpers.image_helper import ImageHelper

from proxy import ModelProxy


class MainView(QMainWindow, View):
    def __init__(self, proxy: ModelProxy, parent=None) -> None:
        super().__init__(parent)
        self.proxy = proxy
        self.score = 0.5
        self.bbox_number = 0
        self.setupUi(self)
        self.set_all()
        self.show()

    def set_all(self) -> None:
        self.select_image_button.clicked.connect(self.select_image)
        self.display_all_check_box.stateChanged.connect(self.display_all)
        self.smallest_score_spinbox.valueChanged.connect(self.score_changed)
        self.bounding_box_number_spinbox.valueChanged.connect(
            self.bounding_boxes_changed
        )

    def __load_image(self, image_path: str) -> None:
        self.proxy.load_image(image_path)
        self.__set_max(self.proxy.get_bbox_number(self.score))
        self.score_changed()

    def __set_max(self, bbox_number: int) -> None:
        self.bounding_box_number_spinbox.setMaximum(bbox_number)

    def __display_image(self, image_path: str) -> None:
        self.image_label.setPixmap(QPixmap(image_path).scaled(self.image_label.size()))

    def __reset(self) -> None:
        self.bbox_number = 0
        self.__set_max(self.proxy.get_bbox_number(self.score) - 1)
        self.bounding_box_number_spinbox.setValue(0)

    def __update(self) -> None:
        if self.display_all_check_box.isChecked():
            self.__reset()
            self.bounding_box_number_spinbox.setDisabled(True)
            self.image = self.proxy.show_all_bboxes(self.score)
        else:
            self.bounding_box_number_spinbox.setDisabled(False)
        self.__display_image(ImageHelper.prepareImage(self.image, self.image_label))

    def select_image(self) -> None:
        image = QFileDialog.getOpenFileName(
            self, "Select Image", os.getcwd(), "Images (*.jpg *.JPEG *.JPG *.png)"
        )[0]
        if image == "":
            return
        self.__load_image(image)

    def display_all(self) -> None:
        self.__update()

    def score_changed(self) -> None:
        self.score = self.smallest_score_spinbox.value()
        self.image = self.proxy.show_certain_bbox(self.bbox_number, self.score)
        self.__reset()
        self.__update()

    def bounding_boxes_changed(self) -> None:
        self.bbox_number = self.bounding_box_number_spinbox.value()
        self.image = self.proxy.show_certain_bbox(self.bbox_number, self.score)
        self.__update()
