"""Discontinued methodology for PIV.
"""
import contextlib

from datetime import datetime

import numpy as np
import scipy.ndimage as ndimage
import joblib

from joblib import Parallel, delayed
from openpiv import tools, pyprocess, scaling, filters,validation, preprocess, piv
from tqdm import tqdm, trange

from ..files import ims


## For joblib progressbar, credit to frenzykryger at stackexchange: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/49950707#49950707
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def load_data(data_path, morphology=True):
    print("Loading data...")
    raw_data = ims.load_video_stack(data_path).squeeze().astype(float)
    
    print("Removing background signal...")
    background_signal = np.mean(raw_data, axis=0)
    raw_data -= (background_signal + 5)

    print("Clipping data...")
    raw_data[raw_data < 0] = 0
    raw_data[raw_data > 20] = 20
    if not morphology:
        return raw_data

    for i in trange(raw_data.shape[0], desc="Preprocessing data"):
        raw_data[i] = ndimage.grey_closing(ndimage.grey_opening(raw_data[i], 3), 3)
    
    return raw_data


def find_framerate__s_per_frame(metadata):
    time_info = metadata['TimeInfo']
    first_time = datetime.fromisoformat(time_info['TimePoint1'])
    last_time = datetime.fromisoformat(time_info[f'TimePoint{len(time_info)-3}'])
    duration = last_time - first_time
    duration__s = duration.seconds - duration.microseconds * 1e-6
    return duration__s / (len(time_info) - 3)


def track_between_frames__px_per_s(start_frame_idx, image_stack, dt, window_size, overlap, search_area_size):
    i = start_frame_idx

    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        image_stack[i], 
        image_stack[i+1], 
        window_size=window_size, 
        dt=dt, 
        overlap=overlap,
        search_area_size=search_area_size,
    )
    
    return u0, v0


def track_particles(
    data_path,
    n_jobs=4,
    window_size=8,
    overlap=4,
    search_area_size=8,
    morphology=True
):
    # Load data
    image_stack = load_data(data_path, morphology=morphology)
    metadata = ims.load_ims_metadata(data_path)
    image_size = ims.find_physical_image_size(metadata)[1:]
    pixel_size = np.round(np.array(image_size) / image_stack.shape[1:], 3)
    n_velocities = image_stack.shape[0] - 1

    # Compute velocities
    piv_args = {
        'image_stack': image_stack, 
        'dt': find_framerate__s_per_frame(metadata),
        'window_size': window_size,
        'overlap': overlap,
        'search_area_size': search_area_size,
    }
    with tqdm_joblib(tqdm(desc="Tracking particles", total=(n_velocities))) as progress_bar:
        velocities__px_per_s = Parallel(n_jobs=n_jobs)(
            delayed(track_between_frames__px_per_s)(i, **piv_args) for i in range(n_velocities)
        )
    
    # Add velocities to separate arrays
    x_vel__px_per_s = np.stack([vel[0] for vel in velocities__px_per_s], axis=0)
    y_vel__px_per_s = np.stack([vel[1] for vel in velocities__px_per_s], axis=0)
    
    # Scale velocities to obtain µm/s
    x_vel__µm_per_s = x_vel__px_per_s * pixel_size[0]
    y_vel__µm_per_s = y_vel__px_per_s * pixel_size[1]
    velocities__µm_per_s = np.stack([x_vel__µm_per_s, y_vel__µm_per_s], axis=-1)
    
    # Find original image coordinates
    coord_x, coord_y = pyprocess.get_coordinates(
        image_size=image_stack[0].shape, 
        search_area_size=search_area_size, 
        overlap=overlap,
    )

    return velocities__µm_per_s, coord_x, coord_y
