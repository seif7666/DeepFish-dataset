import numpy
import cv2

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QLabel


class ImageHelper:
    @staticmethod
    def prepareImage(
        frame: numpy.ndarray,
        label: QLabel,
    ) -> QImage:
        frame = QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QImage.Format_RGB888,
        )
        frame = frame.scaled(label.size())
        return frame
