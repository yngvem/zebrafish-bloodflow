import matplotlib.backends.backend_qt5agg as mpl_backend
import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import pyvista as pv
import pyvistaqt
import vtk
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from skimage import measure

from ..vtk import pyvista_interface


class NavigationToolbar(mpl_backend.NavigationToolbar2QT):
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.remove_tool(1)
        self.remove_tool(1)
        self.remove_tool(4)
        self.remove_tool(4)

    def remove_tool(self, tool_idx):
        self.removeAction(self.actions()[tool_idx])


class MatplotlibView(QtWidgets.QWidget):
    def __init__(self, figure, show_coordinates=True, parent=None, flags=Qt.WindowFlags()):
        super().__init__(parent=parent, flags=flags)
        self.figure = figure
        self.canvas = mpl_backend.FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        if not show_coordinates:
            self.toolbar.remove_tool(len(self.toolbar.actions()) - 1)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.canvas)
        self.layout().addWidget(self.toolbar)
        self.updateGeometry()


class SliceViewer(MatplotlibView):
    def __init__(
        self,
        image,
        axis=0,
        voxel_size=(1, 1, 1),
        figure=None,
        parent=None,
        flags=Qt.WindowFlags()
    ):
        if figure is None:
            figure = plt.Figure(facecolor="black")
        self.axis = axis
        self.image = image
        super().__init__(figure, parent=parent, flags=flags)

        # Setup child widgets
        self.slider = IntSlider(
            min=0,
            max=image.shape[axis]-1,
            description="Slice: ",
            parent=self.toolbar
        )
        
        # Insert slider as second to last element in the toolbar
        coordinates = self.toolbar.actions()[-1]
        self.toolbar.removeAction(coordinates)
        self.toolbar.addWidget(self.slider)
        self.toolbar.addAction(coordinates)

        # 
        self.drawing = figure.add_axes((0, 0, 1, 1))
        self.drawing.format_coord = lambda x, y: f"({x:.0f}, {y:.0f}): "

        pixel_size = [vs for i, vs in enumerate(voxel_size) if i != axis]
        image_size = [s for i, s in enumerate(image.shape) if i != axis]
        aspect = pixel_size[0]/pixel_size[1]
        self.imshow = self.drawing.imshow(image[self.curr_img_slice], vmin=0, vmax=1, aspect=aspect)
        self.slider.observe(self.update_plot)
        self.updateGeometry()
    
    @property
    def curr_img_slice(self):
        slice_ = [slice(None)]*3
        slice_[self.axis] = self.slider.value
        return slice_
    
    def update_plot(self, *args):
        slice_ = self.curr_img_slice        
        self.imshow.set_data(self.image[slice_])
        self.canvas.draw()


class SurfaceViewer(pyvistaqt.QtInteractor, QtWidgets.QWidget):
    def __init__(self, image, name="Vasculature", voxel_size=(1, 1, 1), parent=None,):
        super().__init__(parent=parent)
        self.image_mesh = pyvista_interface.to_pyvista_grid(image, name=name, spacing=voxel_size)
        self.add_scene()
    def add_scene(self):
        self.isovalue_actor = self.add_mesh_isovalue(
            self.image_mesh,
            show_scalar_bar=False,
            compute_normals=True,
            cmap="viridis",
        )
        self.isovalue_actor.GetProperty().SetInterpolationToPBR()
        self.isovalue_actor.GetProperty().SetMetallic(0.2)
        self.isovalue_actor.GetProperty().SetRoughness(0.5)
        self.slices_actor = self.add_mesh_slice_orthogonal(self.image_mesh)

        self.center_light = vtk.vtkLight()
        self.center_light.SetPosition(self.image_mesh.origin)
        self.center_light.SetFocalPoint(0.0, 0.0, 0.0)
        
        for renderer in self.renderers:
            renderer.AddLight(self.center_light) 


class IntSlider(QtWidgets.QWidget):
    def __init__(
        self,
        value=None,
        min=0,
        max=99,
        step=1,
        description='Test:',
        readout=True,
        readout_format='d',
        *,  # Qt parameters as keyword-only argument
        parent=None
    ):
        super().__init__(parent=parent)

        if value is None:
            value = (min + max)//2

        self.callbacks = []
        self.qt_slider = QtWidgets.QSlider(Qt.Horizontal,  self)
        self.qt_slider.valueChanged.connect(self.update_callbacks)
        self.qt_slider.setMinimum(min)
        self.qt_slider.setMaximum(max//step)
        self.step = step

        self.description = description
        self.readout = readout
        self.readout_format = readout_format

        self.qt_label = QtWidgets.QLabel()
        self._update_label(self.value)
        self.callbacks.append(self._update_label)

        self.value = value

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.qt_label)
        self.layout().addWidget(self.qt_slider)
        self.updateGeometry()
    
    @property
    def value(self):
        return self.qt_slider.value()*self.step

    @value.setter
    def value(self, value):
        self.qt_slider.setValue(value//self.step)
    
    def observe(self, func):
        self.callbacks.append(func)
    
    def update_callbacks(self):
        for callback in self.callbacks:
            callback(self.value)
    
    def _update_label(self, value):
        self.qt_label.setText(f"{self.description}{value:{self.readout_format}}")


class FloatSlider(QtWidgets.QWidget):
    def __init__(
        self,
        value=None,
        min=0,
        max=1,
        step=0.1,
        description='Test:',
        readout=True,
        readout_format='f',
        *,  # Qt parameters as keyword-only argument
        parent=None
    ):
        super().__init__(parent=parent)

        if value is None:
            value = (min + max)/2

        self.qt_slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.qt_slider.valueChanged.connect(self.update_callbacks)
        self.qt_slider.setMinimum(0)
        self.qt_slider.setMaximum(int((max - min)/step))
        self._step = step
        self._min = min
        self._max = max

        self.description = description
        self.readout = readout
        self.readout_format = readout_format

        self.qt_label = QtWidgets.QLabel()
        self._update_label(self.value)
        self.callbacks = [self._update_label]

        self.value = value

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.qt_label)
        self.layout().addWidget(self.qt_slider)
        self.updateGeometry()

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, new_min):
        value = self.value
        self._min = new_min
        self.qt_slider.setMaximum(int((self.max - self.min)/self.step))
        self.value = value

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, new_max):
        value = self.value
        self._max = new_max
        self.qt_slider.setMaximum(int((self.max - self.min)/self.step))
        self.value = value

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, new_step):
        value = self.value
        self._step = new_step
        self.qt_slider.setMaximum(int((self.max - self.min)/self.step))
        self.value = value

    @property
    def value(self):
        return self.min + self.qt_slider.value()*self.step

    @value.setter
    def value(self, value):
        offset = int((value - self.min)/self.step)
        self.qt_slider.setValue(offset)

    def observe(self, func):
        self.callbacks.append(func)
    
    def update_callbacks(self):
        for callback in self.callbacks:
            callback(self.value)
    
    def _update_label(self, value):
        self.qt_label.setText(f"{self.description}{value:{self.readout_format}}")    

