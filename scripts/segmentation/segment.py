from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import morphsnakes
import numpy as np
from scipy import ndimage
from skimage import exposure, io, filters
from tqdm import tqdm, trange
from matplotlib.animation import FuncAnimation
import medvis
import matplotlib.cm as cm

from confocal_microscopy.files import ims
from confocal_microscopy.filters import anisotropic_diffusion
import confocal_microscopy.filters.exposure as cm_exposure
import confocal_microscopy.filters as cm_filters
from confocal_microscopy.plotting import implay
from confocal_microscopy.utils import Pipeline, PreviousPipelineValue, apply_slicewise
from confocal_microscopy.mask import filter_small_regions
from confocal_microscopy.vtk import pyvista_interface

log = []
def make_progress_bar(num_its):
    it = iter(trange((num_its+1)))
    def callback(contours):
        global log
        next(it)
        log.append(contours.copy())

    return callback


if __name__ == "__main__":
    image_path = Path("/home/yngve/Documents/Fish 1 complete/Cancer region/Blood vessels 3d stack/fast_2020-09-02_Federico s_10.41.10_JFM9CC2.ims")
    image = ims.load_image_stack(image_path, resolution_level=2)

    use_checkpoint = True
    pipeline = Pipeline(image, "pipeline")
    pipeline.add_step(
        f=cm_exposure.normalise,
        name="Normalisation",
        use_checkpoint=use_checkpoint
    )
    pipeline.add_step(
        f=cm_exposure.reduce_dynamic_range,
        name="Dynamic range reduction",
        kwargs=dict(range_min=1, range_max=99),
        use_checkpoint=use_checkpoint,
    )
    #pipeline.add_step(
    #    f=exposure.equalize_adapthist,
    #    name="Adaptive equalisation",
    #    use_checkpoint=use_checkpoint
    #)
    pipeline.add_step(
        f=anisotropic_diffusion,
        name="Anisotropic diffusion",
        kwargs=dict(edge_scale=0.01, step_size=0.8, num_steps=20, progress=True),
        use_checkpoint=use_checkpoint
    )
    pipeline.add_step(
        f=apply_slicewise(ndimage.median_filter, progress=True),
        name="Median filter",
        kwargs=dict(size=3),
        use_checkpoint=use_checkpoint
    )
    pipeline.add_step(
        f=cm_exposure.normalise,
        name="Second normalisation",
        use_checkpoint=use_checkpoint
    )
    pipeline.add_step(
        f=lambda x: (x > 0.3).astype(float),
        name="Threshold",
        use_checkpoint=use_checkpoint
    )
    pipeline.add_step(
        f=morphsnakes.morphological_chan_vese,
        name="snakes1",
        input_name="Second normalisation",
        kwargs=dict(
            iterations=1,
            lambda1=1,
            smoothing=10,
            lambda2=5,
            init_level_set=PreviousPipelineValue("Threshold")
        ),
        use_checkpoint=use_checkpoint
    )
    pipeline.add_step(
        f=morphsnakes.morphological_chan_vese,
        name="snakes2",
        input_name="Second normalisation",
        kwargs=dict(
            iterations=1,
            lambda1=1,
            smoothing=10,
            lambda2=5,
            init_level_set=PreviousPipelineValue("snakes1")
        ),
        use_checkpoint=use_checkpoint
    )
    pipeline.add_step(
        f=morphsnakes.morphological_chan_vese,
        name="snakes3",
        input_name="Second normalisation",
        kwargs=dict(
            iterations=1,
            lambda1=1,
            smoothing=10,
            lambda2=5,
            init_level_set=PreviousPipelineValue("snakes2")
        ),
        use_checkpoint=use_checkpoint
    )
    pipeline.add_step(
        f=morphsnakes.morphological_chan_vese,
        name="snakes4",
        input_name="Second normalisation",
        kwargs=dict(
            iterations=1,
            lambda1=1,
            smoothing=10,
            lambda2=5,
            init_level_set=PreviousPipelineValue("snakes3")
        ),
        use_checkpoint=use_checkpoint
    )
    pipeline.add_step(
        f=filter_small_regions,
        name="Remove speckles",
        kwargs=dict(
            min_size=1000
        ),
        use_checkpoint=False
    )
    pipeline.add_step(
        f=apply_slicewise(ndimage.binary_closing, progress=True),
        name="Closing",
        kwargs=dict(
            iterations=5
        ),
        use_checkpoint=False
    )
    pipeline.add_step(
        f=lambda x: x.astype(float),
        name="Casting",
        use_checkpoint=False
    )
    pipeline.add_step(
        f=ndimage.zoom,
        name="zoom",
        kwargs=dict(zoom=(2.5, 1, 1)),
        use_checkpoint=False
    )
    pipeline.add_step(
        f=lambda x: ndimage.gaussian_filter(x - 0.5, sigma=1),
        name="Smoothing",
        use_checkpoint=False
    )
    num_labels = ndimage.label(pipeline.current_image)[1]
    print(f"Det er totalt {num_labels} strukturer")
    def label(*args, **kwargs):
        return ndimage.label(*args, **kwargs)[0]


    pipeline.save_step("Remove speckles", scale=65535, dtype=np.uint16)
    pipeline.save_step("Casting", scale=65535, dtype=np.uint16)
    pipeline.save_step("Second normalisation", scale=65535, dtype=np.uint16)
    pipeline.save_step("Normalisation", scale=65535, dtype=np.uint16)
    
    contour = apply_slicewise(medvis.create_outline)(pipeline.current_image, colour="tomato")
    image = cm.viridis(pipeline.images["Second normalisation"])

    alpha = 1
    implay(pipeline.images["Second normalisation"])
    plt.show()


    pyvista_interface.to_pyvista_grid(pipeline.current_image).save("mask.vti")