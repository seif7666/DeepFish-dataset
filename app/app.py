from PySide6.QtWidgets import QApplication
from src.presentation.views.logic.main_view import MainView

if __name__ == "__main__":
    import sys
    
    app = QApplication(sys.argv)
    window = MainView()
    sys.exit(app.exec())