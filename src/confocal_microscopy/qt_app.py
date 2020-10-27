from PyQt5 import QtWidgets
import os
import numpy as np
from numpy import cos
from mayavi import mlab
from confocal_microscopy.files.ims import load_image_stack
from scipy.ndimage import zoom
from skimage.filters import threshold_otsu

os.environ['ETS_TOOLKIT'] = 'qt4'
from pyface.qt import QtGui, QtCore  # noqa: E402
from traits.api import HasTraits, Instance, on_trait_change  # noqa: E402
from traits.trait_numeric import Array  # noqa: E402
from traitsui.api import View, Item  # noqa: E402
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor  # noqa: E402


## create Mayavi Widget and show

class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    image = Array()

    @on_trait_change('scene.activated')
    def initiate_plot(self):
    ## PLot to Show
        self.scene.background = (0, 0, 0)
        print("Generating slicers")
        self.x_slice = mlab.volume_slice(self.image, colormap="magma", plane_orientation='x_axes')
        self.y_slice = mlab.volume_slice(self.image, colormap="magma", plane_orientation='y_axes')
        self.z_slice = mlab.volume_slice(self.image, colormap="magma", plane_orientation='z_axes')
        th = threshold_otsu(self.image)
        print("Generating contours")
        self.contour = mlab.contour3d((self.image > th).astype(float))

    view = View(
        Item(
            'scene',
            editor=SceneEditor(scene_class=MayaviScene),
            height=250,
            width=300,
            show_label=False
        ),
        resizable=True
    )


class MayaviQWidget(QtGui.QWidget):
    def __init__(self, image, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.visualization = Visualization()
        self.visualization.image = image

        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)


#### PyQt5 GUI ####
class Ui_MainWindow(object):
    def setupUi(self, image, MainWindow):
    ## MAIN WINDOW
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(200, 200, 1100, 700)

    ## CENTRAL WIDGET
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

    ## GRID LAYOUT
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

    ## BUTTONS
        self.button_default = QtWidgets.QPushButton(self.centralwidget)
        self.button_default.setObjectName("button_default")
        self.gridLayout.addWidget(self.button_default, 0, 0, 1, 1)

    ## Mayavi Widget 1    
        container = QtGui.QWidget()
        self.mayavi_widget1 = MayaviQWidget(image, container)
        self.gridLayout.addWidget(self.mayavi_widget1, 1, 0, 2, 1)

    ## SET TEXT 
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "VasculatureViewer"))
        self.button_default.setText(_translate("MainWindow", "Default Values"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    MainWindow = QtWidgets.QMainWindow()

    print("Loading image...")
    image = load_image_stack("/home/yngve/Documents/Fish 1 complete/Cancer region/Blood vessels 3d stack/fast_2020-09-02_Federico s_10.41.10_JFM9CC2.ims", resolution_level=2)
    image = zoom(image, (1, 1, 1), order=1)
    print("Loaded image")
    ui = Ui_MainWindow()
    ui.setupUi(image, MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
