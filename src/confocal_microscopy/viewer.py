import os

os.environ['ETS_TOOLKIT'] = 'qt4'

import mayavi.mlab as mlab
import numpy as np
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from pyface.qt import QtCore, QtGui
from PyQt5 import QtWidgets
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import Item, View


class Viewer(HasTraits):
    scene = Instance(MlabSceneModel, ())):
    
    def __init__(self, image):
        self.image = image
        self.figure = None
    
    def visualise(self):
        self.figure = mlab.figure(bgcolor=(0, 0, 0))

        mlab.volume_slice(self.image, colormap="magma", plane_orientation='x_axes', figure=self.figure)
        mlab.volume_slice(self.image, colormap="magma", plane_orientation='y_axes', figure=self.figure)
        mlab.volume_slice(self.image, colormap="magma", plane_orientation='z_axes', figure=self.figure)


