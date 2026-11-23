import sys
from PyQt6.QtWidgets import QApplication

from Annotation_App.processing import VentilationProcessor
from Annotation_App.annotations import AnnotationStore
from Annotation_App.ui_window import VentilationAnnotatorWindow


def main():
    fs = 128
    win_sec = 5.0
    step_sec = 2.0
    max_duration_sec = 3600.0  # Load at most the last 1 hour

    app = QApplication(sys.argv)

    processor = VentilationProcessor(fs=fs, max_duration_sec=max_duration_sec)
    store = AnnotationStore(
        fs=fs,
        win_sec=win_sec,
        step_sec=step_sec,
        max_duration_sec=max_duration_sec,
    )
    window = VentilationAnnotatorWindow(processor=processor, store=store)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()