import os
os.environ['ETS_TOOLKIT'] = 'qt4'

import numpy as np
from PyQt5 import  QtWidgets
from pyface.qt import QtGui, QtCore
from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

import mayavi.mlab as mlab


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


