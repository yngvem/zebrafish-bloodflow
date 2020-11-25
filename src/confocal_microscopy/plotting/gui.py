import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

import numexpr as ne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt5agg as mpl_backend
import vtk


import pyvista as pv
import pyvistaqt

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
        self.isovalue_actor, self.isovalue_algorithm = self.add_mesh_isovalue(
            self.image_mesh,
            show_scalar_bar=False,
            compute_normals=True,
            cmap="viridis",
        )
        self.isovalue_actor.GetProperty().SetInterpolationToPBR()
        self.isovalue_actor.GetProperty().SetMetallic(0.2)
        self.isovalue_actor.GetProperty().SetRoughness(0.5)

        self.center_light = vtk.vtkLight()
        self.center_light.SetPosition(self.image_mesh.origin)
        self.center_light.SetFocalPoint(0.0, 0.0, 0.0)
        
        for renderer in self.renderers:
            renderer.AddLight(self.center_light) 


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, image, voxel_size=(1, 1, 1), parent=None, flags=Qt.WindowFlags()):
        super().__init__(parent, flags)
        self.button = QtWidgets.QPushButton(text="Hei")
        self.button.clicked.connect(self.update_img)
        self.image = np.asfortranarray(image)
        self.mpl_views = [None]*3
        self.mpl_views[0] = SliceViewer(self.image, voxel_size=voxel_size, axis=0, parent=self)
        self.mpl_views[1] = SliceViewer(self.image, voxel_size=voxel_size, axis=1, parent=self)
        self.mpl_views[2] = SliceViewer(self.image, voxel_size=voxel_size, axis=2, parent=self)
        self.surface_viewer = SurfaceViewer(self.image, voxel_size=voxel_size, parent=self)
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        self.inner_layout = None
        self.set_grid_layout()
        self.surface_viewer.mouseDoubleClickEvent = lambda x: ImageViewer.mouseDoubleClickEvent(self, x)
        for view in self.mpl_views:
            view.canvas.mouseDoubleClickEvent = lambda x: ImageViewer.mouseDoubleClickEvent(self, x)
        self.single_view = False
    
    def mouseDoubleClickEvent(self, event):
        print(self.childAt(event.pos()))
        if self.single_view:
            self.set_grid_layout()
            self.single_view = False
        else:
            self.set_central_layout(self.childAt(event.pos()))
            self.single_view = True

    def set_grid_layout(self):
        if self.inner_layout is not None:
            self.layout().removeItem(self.inner_layout)
        self.inner_layout = QtWidgets.QGridLayout()        
        self.inner_layout.addWidget(self.mpl_views[0], 0, 0)
        self.inner_layout.addWidget(self.mpl_views[1], 0, 1)
        self.inner_layout.addWidget(self.mpl_views[2], 1, 1)
        self.inner_layout.addWidget(self.surface_viewer, 1, 0)
        self.inner_layout.addWidget(self.button, 2, 2)
        self.layout().addLayout(self.inner_layout)
        self.updateGeometry()

    def set_central_layout(self, widget):
        if self.inner_layout is not None:
            self.layout().removeItem(self.inner_layout)
        self.inner_layout = QtWidgets.QGridLayout()
        self.inner_layout.addWidget(widget, 0, 0)
        self.layout().addLayout(self.inner_layout)
        self.updateGeometry()
    
    def update_plots(self, *args):
        for view in self.mpl_views:
            view.update_plot()
    
    def update_img(self, *args):
        self.image *= 2
        self.image[self.image > 1] = 1
        #self.surface_viewer.image_mesh.point_arrays["Vasculature"] = self.image.flatten('F')
        self.update_plots()
        self.surface_viewer.isovalue_algorithm.Update()
        self.surface_viewer.isovalue_actor.shallow_copy(self.surface_viewer.isovalue_algorithm.GetOutput())
        for renderer in self.surface_viewer.renderers:
            #renderer.RemoveAllViewProps()

            renderer.Render()
        #self.surface_viewer.add_scene()
        print(len(self.surface_viewer.renderers))
        print(np.mean(self.surface_viewer.image_mesh.point_arrays["Vasculature"]))
        print(np.mean(self.image))



class Transformer(ImageViewer):
    def __init__(self, image, voxel_size=(1, 1, 1), parent=None, flags=Qt.WindowFlags()):
        super().__init__(image, voxel_size, parent, flags)
        self.input_image = image.copy()
        self._image = image
    
    @property
    def image(self):
        return self._image

    def transform(self) -> None:
        """Modify self.image_stack inplace.
        """
        pass


class Histogram(Transformer):
    def __init__(self, image, voxel_size=(1, 1, 1), parent=None, flags=Qt.WindowFlags()):
        super().__init__(
            image=image,
            voxel_size=voxel_size,
            parent=parent,
            flags=flags
        )

        widget = QtWidgets.QWidget(parent=self)
        layout = QtWidgets.QVBoxLayout()
        self.min = FloatSlider(
            0,
            min=0,
            max=1,
            step=0.001,
            description="Minimum value:",
            parent=widget
        )
        self.max = FloatSlider(
            1,
            min=0,
            max=1,
            step=0.001,
            description="Maximum value:",
            parent=widget
        )
        self.min.observe(self.transform)
        self.max.observe(self.transform)
        self.min.observe(self.update_plots)
        self.max.observe(self.update_plots)
        layout.addWidget(self.min)
        layout.addWidget(self.max)
        widget.setLayout(layout)
        self.layout().addWidget(widget, 1, 0)
    
    def transform(self, *args):
        image = self.input_image
        min_ = self.min.value
        max_ = self.max.value
        out = self.image
        ne.evaluate("(image - min_)/(max_ - min_)", out=out)

        self.image[self.image < 0] = 0
        self.image[self.image > 1] = 1


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



if __name__ == "__main__":
    from confocal_microscopy.files import ims
    from pathlib import Path
    from scipy import ndimage

    image_path = Path("/home/yngve/Documents/Fish 1 complete/Cancer region/Blood vessels 3d stack/fast_2020-09-02_Federico s_10.41.10_JFM9CC2.ims")
    image = ims.load_image_stack(image_path, resolution_level=2).astype(float)
    image -= image.min() 
    image /= image.max()


    app = QtWidgets.QApplication(sys.argv)
    main = ImageViewer(image, voxel_size=(1000, 408, 408))
    main.show()
    sys.exit(app.exec_())