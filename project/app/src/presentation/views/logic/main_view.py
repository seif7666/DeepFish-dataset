import os
from PySide6.QtWidgets import QMainWindow, QFileDialog
from PySide6.QtGui import QPixmap
from app.src.presentation.views.ui.main_view import Ui_deep_fish_window as View

from proxy import ModelProxy


class MainView(QMainWindow, View):
    def __init__(self, proxy: ModelProxy, parent=None) -> None:
        super().__init__(parent)
        self.proxy = ModelProxy()
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

    def select_image(self) -> None:
        image = QFileDialog.getOpenFileName(
            self, "Select Image", os.getcwd(), "Images (*.jpg *.JPEG *.JPG *.png)"
        )[0]
        if image == "":
            return
        self.proxy.load_image(image)
        self.__display_image(image)

    def display_all(self) -> None:
        if self.display_all_check_box.isChecked():
            pass  # TODO: Display All Boxes

    def score_changed(self) -> None:
        pass  # TODO: Change Score and Display Image

    def bounding_boxes_changed(self) -> None:
        pass  # TODO: Change boxes Number and Display Image

    def __display_image(self, image_path: str) -> None:
        self.image_label.setPixmap(QPixmap(image_path).scaled(self.image_label.size()))
