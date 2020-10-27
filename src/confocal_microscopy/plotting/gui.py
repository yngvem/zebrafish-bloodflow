import sys

## namespace organization changed in PyQt5 but the class name was kept.
## importing this way makes it easier to change to PyQt5 later
from PyQt5.QtWidgets import (QMainWindow, QApplication, QDockWidget, QWidget, QGridLayout, QSlider)
from PyQt5.QtCore import Qt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5agg


class MainWindow(QMainWindow):
    x = np.arange(0, 10, 0.1)
    cos = 0
    sin = 0

    def __init__(self):
        super().__init__()

        self.figure = plt.figure()
        self.drawing = self.figure.add_subplot(111)
        self.canvas = matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg(self.figure)

        self.setCentralWidget(self.canvas)

        dock = QDockWidget("Values")
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        sliders = QWidget()
        sliders_grid = QGridLayout(sliders)

        def add_slider(foo, col):
            sld = QSlider(Qt.Vertical, sliders)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.valueChanged[int].connect(foo)
            sld.valueChanged.connect(self.plot)
            sliders_grid.addWidget(sld, 0, col)

        add_slider(foo=self.set_cos, col=0)
        add_slider(foo=self.set_sin, col=1)

        dock.setWidget(sliders)

        self.plot()

    def set_cos (self, v):
        self.cos = v / 100

    def set_sin (self, v):
        self.sin = v / 100

    def plot (self):
        self.drawing.clear()
        s = np.sin(self.x + self.sin)
        c = np.cos(self.x + self.cos)
        self.drawing.plot(self.x, s, 'r', self.x, c, 'r', self.x, s + c, 'b')
        self.drawing.set_ylim(-2, 2)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())