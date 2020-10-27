import pyvista as pv
import numpy as np


def to_pyvista_grid(image_stack, name="Scalars_", spacing=(1, 1, 1)):
    image = pv.UniformGrid()
    image.dimensions = np.array(image_stack.shape)
    image.spacing = spacing
    image.point_arrays[name] = image_stack.flatten('F')
    return image