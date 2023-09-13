import os
from PySide6.QtWidgets import QMainWindow, QFileDialog
from src.presentation.views.ui.main_view import Ui_deep_fish_widnow as View


class MainView(QMainWindow, View):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self.set_all()
        self.show()

    def set_all(self) -> None:
        self.select_image_button.clicked.connect(self.select_image)
        self.display_all_check_box.stateChanged.connect(self.display_all)
        self.smallest_score_spinbox.valueChanged.connect(self.score_changed)
        self.bounding_box_number_spinbox.valueChanged.connect(self.bounding_boxes_changed)

    def select_image(self) -> None:
        image = QFileDialog.getOpenFileName(self, "Select Image", os.getcwd(), "Images (*.jpg *.JPEG *.JPG *.png)")[0]
        if image == "":
            return
        # TODO: Get Image to model

    def display_all(self) -> None:
        if self.display_all_check_box.isChecked():
            pass # TODO: Display All Boxes

    def score_changed(self) -> None:
        pass # TODO: Change Score and Display Image

    def bounding_boxes_changed(self) -> None:
        pass # TODO: Change boxes Number and Display Image