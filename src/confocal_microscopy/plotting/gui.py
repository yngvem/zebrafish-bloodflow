import numexpr as ne
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from .gui_components import FloatSlider, IntSlider, SliceViewer, SurfaceViewer


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


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from scipy import ndimage

    from confocal_microscopy.files import ims

    image_path = Path("/home/yngve/Documents/Fish 1 complete/Cancer region/Blood vessels 3d stack/fast_2020-09-02_Federico s_10.41.10_JFM9CC2.ims")
    image = ims.load_image_stack(image_path, resolution_level=2).astype(float)
    image -= image.min() 
    image /= image.max()


    app = QtWidgets.QApplication(sys.argv)
    main = ImageViewer(image, voxel_size=(1000, 408, 408))
    main.show()
    sys.exit(app.exec_())