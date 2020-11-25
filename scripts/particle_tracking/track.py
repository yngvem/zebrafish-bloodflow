from pathlib import Path
import warnings
import time
import argparse

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

tp.enable_numba()
tp.quiet()


class IMSLoader(ims.LazyIMSVideoLoader):
    def _preprocess(self, frame):
        frame = frame - self.background_signal
        frame[frame < 0] = 0
        frame = ndimage.grey_opening(frame, 3)
        frame = ndimage.grey_closing(frame, 5)
        frame -= self._limits[0]
        frame /= self._limits[1]
        frame *= 255
        frame = frame.astype(np.uint8)
        return frame


def track_particles(path, preprocess=True):
    path = Path(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with IMSLoader(path, limits=(0, 20)) as imsloader:
            print("Finding blobs...", flush=True)
            features = tp.batch(imsloader, 5, preprocess=False)

    print("Linking blobs between frames...", flush=True)
    features = tp.link(features, 10, memory=2, adaptive_step=1)

    print("Filtering out blobs that didn't stay for long...", flush=True)
    features = tp.filter_stubs(features, 5)
    print(f"Found {len(features['particle'].unique())} tracks.")

    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    parser.add_argument("num_tasks", type=int)
    args = parser.parse_args()
    task_id = args.task_id
    num_tasks = args.num_tasks
    assert 0 <= task_id and task_id < num_tasks

    parent = Path("/home/yngve/Documents/Fish 1 complete/")
    parent = Path("/media/yngve/TOSHIBA EXT (YNGVE)/7 DAY OLD Fish without tumors/")
    files = list(parent.glob("**/*red ch_*.ims"))
    num_digits = len(str(len(files)))
    PREPROCESS = False
    failed = {}
    for i, path in enumerate(files):
        if i % num_tasks != task_id:
            continue
        message = f"Track {i+1:{num_digits}d} out of {len(files):{num_digits}d}:"
        print(message)
        print("="*len(message))
        out_path = path.parent / f"{path.stem}.csv"
        failed_path = path.parent / f"{path.stem}_failed"
        start_time = time.time()
        if out_path.is_file() or failed_path.is_file():
            print("Already completed")
            continue
        try:
            tracks = track_particles(path, preprocess=PREPROCESS)
        except OSError:
            print(f"Failed opening file at {path}!")
            failed[path] = "OSError"
        except tp.linking.utils.SubnetOversizeException:
            print(f"Failed linking at {path}!")
            with failed_path.open("w") as f:
                f.write("")
            failed[path] = "SubnetOversizeException"
        else:
            stop_time = time.time()
            duration = stop_time - start_time
            print(f"Finished tracking, took {duration:.0f} s")
            tracks.to_csv(out_path)
            print(f"Saved tracks: {out_path}")

    print("These files were corrupt or failed:")
    for path, reason in failed.items:
        print(f"Failed {path} as consequence of {reason}")
