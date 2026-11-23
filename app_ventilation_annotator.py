import sys
from PyQt6.QtWidgets import QApplication

from Annotation_App.processing import VentilationProcessor
from Annotation_App.annotations import AnnotationStore
from Annotation_App.ui_window import VentilationAnnotatorWindow


def main():
    fs = 128
    win_sec = 5.0
    step_sec = 2.0

    app = QApplication(sys.argv)

    processor = VentilationProcessor(fs=fs)
    store = AnnotationStore(fs=fs, win_sec=win_sec, step_sec=step_sec)
    window = VentilationAnnotatorWindow(processor=processor, store=store)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()