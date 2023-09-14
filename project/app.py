from PySide6.QtWidgets import QApplication
from app.src.presentation.views.logic.main_view import MainView

from proxy import ModelProxy

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainView(proxy=ModelProxy())
    sys.exit(app.exec())
