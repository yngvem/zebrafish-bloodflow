import json
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets
from scipy import ndimage
from tqdm import tqdm

import confocal_microscopy.roi_tools.centerline as centerline_tools
from confocal_microscopy.tracking.utils import load_background


class ROIExtractor(QtWidgets.QWidget):
    def __init__(self, background_path, parent=None):
        super().__init__(parent=parent)
        self.background = load_background(background_path)
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_axes((0, 0, 1, 1))
        self.ax.axis('off')
        self.imshow = self.ax.imshow(self.background)
        
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.vertices = {'x': [], 'y': []}
        self.all_vertices = [self.vertices]

        self.all_centerlines = []

        self.lineplot = self.ax.plot([], [], color='tomato')[0]
        self.all_lineplots = [self.lineplot]

        self.scatterplot = self.ax.plot([], [], 'o', color='tomato')[0]
        self.all_scatterplots = [self.scatterplot]

        self.next_lineplot = self.ax.plot([], [], ':', color='tomato')[0]

        self.press_listener_id = self.canvas.mpl_connect('button_press_event', self.add_vertex)
        self.movement_listener_id = self.canvas.mpl_connect('motion_notify_event', self.view_next_vertex)

    def update_plots(self):
        self.lineplot.set_data(self.vertices['x'], self.vertices['y'])
        self.scatterplot.set_data(self.vertices['x'], self.vertices['y'])
        self.canvas.draw()

    def add_vertex(self, event):
        if event.button == MouseButton.LEFT:
            self.vertices['x'].append(event.xdata)
            self.vertices['y'].append(event.ydata)
        elif event.button == MouseButton.RIGHT:
            self.vertices['x'].pop()
            self.vertices['y'].pop()
            self.view_next_vertex(event)
        self.update_plots()

    def view_next_vertex(self, event):
        if len(self.vertices['x']) > 0:
            self.next_lineplot.set_data(
                [self.vertices['x'][-1], event.xdata],
                [self.vertices['y'][-1], event.ydata]
            )
        else:
            self.next_lineplot.set_data(
                [], []
            )
        self.canvas.draw()
        
    def keyPressEvent(self, event):
        if (event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter):
            if len(self.vertices['x']) == 0:
                return
            self.finish_polygon()
        elif event.key() == QtCore.Qt.Key_Z:
            if len(self.vertices['x']) == 0:
                self.lineplot.set_data([], [])
                self.scatterplot.set_data([], [])
                self.vertices = self.all_vertices.pop()
                self.vertices = self.all_vertices[-1]
                self.lineplot = self.all_lineplots.pop()
                self.lineplot = self.all_lineplots[-1]
                self.all_centerlines.pop()
                self.scatterplot = self.all_scatterplots.pop()
                self.scatterplot = self.all_scatterplots[-1]

                self.vertices['x'].pop()
                self.vertices['y'].pop()
                self.lineplot.set_color('tomato')
                self.scatterplot.set_color('tomato')
                self.update_plots()
            else:
                self.vertices['x'].pop()
                self.vertices['y'].pop()
                self.update_plots()
        elif event.key() == QtCore.Qt.Key_Q:
            if input("Exit without saving current frame? y/[n]").lower() == "y":
                sys.exit(0)
        elif event.key() == QtCore.Qt.Key_Escape:
            self.all_vertices.pop()
            self.all_lineplots.pop()
            self.all_scatterplots.pop()
            self.close()

    def find_centerline(self):
        roi, centerline = centerline_tools.find_centerline_and_clip_roi(self.vertices, self.background.shape)
        self.vertices['x'] = roi['x']
        self.vertices['y'] = roi['y']
        self.all_centerlines.append({'x': centerline[:, 0].tolist(), 'y': centerline[:, 1].tolist()})

        self.ax.scatter(
            centerline[:, 0],
            centerline[:, 1],
            s=5,
            color=cm.inferno(np.arange(len(centerline))/len(centerline))
        )
        self.update_plots()

    def finish_polygon(self):
        self.find_centerline()

        self.lineplot.set_data(
            self.vertices['x'],
            self.vertices['y'],
        )
        self.scatterplot.set_data(
            self.vertices['x'],
            self.vertices['y'],
        )
        
        self.vertices = {'x': [], 'y': []}
        self.all_vertices.append(self.vertices)
        self.lineplot.set_color("white")
        self.scatterplot.set_color("white")
        self.canvas.draw()

        self.lineplot = self.ax.plot([], [], color='tomato')[0]
        self.all_lineplots.append(self.lineplot)

        self.scatterplot = self.ax.plot([], [], 'o', color='tomato')[0]
        self.all_scatterplots.append(self.scatterplot)
    
    def show(self, *args, **kwargs):
        self.showMaximized()
        super().show(*args, **kwargs)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    fish_path = Path("/media/yngve/TOSHIBA EXT (YNGVE)/fish_data/organised/7 DAY OLD Fish without tumors/")

    for background_path in tqdm(sorted(fish_path.glob("Fish */**/*Snap*.ims"))):
        print(fish_path)
        vertex_file = background_path.parent/f"{background_path.stem}_vertices.json"
        if vertex_file.is_file():
            skip = "aaa"
            while skip.lower().strip() not in {"y", "n", ""}:

                skip = input(f"Vertex file: \"{vertex_file}\" exists, skip? ([y]/n)")
            if skip.lower().strip() == "y" or skip.lower().strip() == "":
                continue

        main = ROIExtractor(background_path)
        main.show()
        app.exec_()
        with vertex_file.open("w") as f:
            json.dump({'vertices': main.all_vertices, 'centerlines': main.all_centerlines, 'image_shape': list(main.background.shape)}, f)
