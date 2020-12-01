from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import pims
import trackpy as tp
from skimage.io import imread
from scipy import ndimage
from tqdm import tqdm, trange

from confocal_microscopy.utils import apply_slicewise
from confocal_microscopy.files import ims
from confocal_microscopy.tracking.utils import load_background

tp.enable_numba()
tp.quiet()


def load_background_new_data(path):
    if ("400" not in path.parent.name) and ("1000" not in path.parent.name):
        prefix = path.stem.split("_Federico")[0]
        try:
            background_path = next(path.parent.glob(f"{prefix}_Snap*.ims"))
        except StopIteration:
            return None
    else:
        background_path = next(path.parent.parent.glob("*Snap*.ims"))
    background_image = load_background(background_path)
    return background_image


def load_background_old_data(path):
    background_path = next(path.parent.glob("*Snap*.ims"))
    background_image = load_background(background_path)
    return background_image


def plot_trajectories(path, background_image):
    trajectories = pd.read_csv(path)

    aspect = background_image.shape[1] / background_image.shape[0]
    fig, ax = plt.subplots(dpi=200, figsize=(10, 14/aspect))
    ax.imshow(background_image)
    fig.savefig(path.parent/f"{path.stem}_background.png")
    tp.plot_traj(trajectories, ax=ax)
    fig.savefig(path.parent/f"{path.stem}_min_05_new_preprocess.png")

    ax.clear()
    trajectories = tp.filter_stubs(trajectories, 10)
    ax.imshow(background_image)
    tp.plot_traj(trajectories, ax=ax)
    fig.savefig(path.parent/f"{path.stem}_min_10_new_preprocess.png")

    ax.clear()
    trajectories = tp.filter_stubs(trajectories, 15)
    ax.imshow(background_image)
    tp.plot_traj(trajectories, ax=ax)
    fig.savefig(path.parent/f"{path.stem}_min_15_new_preprocess.png")
    plt.close(fig)

    print(f"Created figures for {path}")


def plot_trajectories_new_data(path):
    background_image = load_background_new_data(path)
    if background_image is None:
        return
    if (path.parent/f"{path.stem}_background.png").is_file() and (path.parent/f"{path.stem}_min_15_new_preprocess.png").is_file():
        return
    plot_trajectories(path, background_image)


def plot_trajectories_old_data(path):
    background_image = load_background_old_data(path)
    if background_image is None:
        return
    if (path.parent/f"{path.stem}_background.png").is_file() and (path.parent/f"{path.stem}_min_15_new_preprocess.png").is_file():
        return
    plot_trajectories(path, background_image)


if __name__ == "__main__":
    parent = Path("/home/yngve/Documents/Fish 1 complete/")
    #parent = Path("/media/yngve/TOSHIBA EXT (YNGVE)/7 DAY OLD Fish without tumors/")
    parent = Path("/media/yngve/TOSHIBA EXT (YNGVE)/fish_data/organised/7 DAY OLD Fish without tumors/")
    files = list(parent.glob("**/*red ch_*.csv"))
    num_digits = len(str(len(files)))
    PREPROCESS = False
    failed = {}
    for path in tqdm(files):
        try:
            plot_trajectories_new_data(path)
        except Exception as e:
            failed['path'] = e
            print(f"Failed {path} because of {e}")

    for path, reason in failed.items():
        print(path, reason)
    