"""Iterate over all video-files in file tree and track particles.


"""

import re
import argparse
import time
import warnings
from pathlib import Path

import trackpy as tp
from scipy import ndimage
import docx

from confocal_microscopy.files import ims

tp.enable_numba()
tp.quiet()


class IMSLoader(ims.LazyIMSVideoLoader):
    def _preprocess(self, frame):
        frame = frame.astype(float)
        
        # Remove background signal
        frame = frame - self.background_signal
        frame[frame < 0] = 0

        # Morphological denoising
        frame = ndimage.grey_opening(frame, 3)
        frame = ndimage.grey_closing(frame, 5)

        # Clip dynamic range
        frame -= self._limits[0]
        frame /= self._limits[1]
        frame *= 255
        frame[frame > 255] = 255
        frame[frame < 0] = 0

        return frame


def track_particles(path):
    """Find particles, link tracks and remove particles that are only present for one frame
    """
    path = Path(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with IMSLoader(path, limits=(2, 50)) as imsloader:
            print("Finding blobs...", flush=True)
            features = tp.batch(imsloader, 5, minmass=50, preprocess=False)

    print("Linking blobs between frames...", flush=True)
    features = tp.link(features, 16, memory=2, adaptive_step=1)

    print("Filtering out blobs that didn't stay for long...", flush=True)
    features = tp.filter_stubs(features, 2)
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
    parent = Path("/media/yngve/TOSHIBA EXT (YNGVE)/fish_data/organised/7 DAY OLD Fish without tumors/")
    parent = Path("/media/yngve/TOSHIBA EXT (YNGVE)/fish_data/organised/")
    files = list(parent.glob("**/*red ch_*.ims"))
    num_digits = len(str(len(files)))
    failed = {}

    for i, path in enumerate(files):
        if i % num_tasks != task_id:
            continue
        message = f"Track {i+1:{num_digits}d} out of {len(files):{num_digits}d}:"
        print(message)
        print("="*len(message))

        # Extract wavelength of current track
        wavelength = None
        if "1000" in path.parent.name:
            wavelength = 1000
        elif "400" in path.parent.name:
            wavelength = 400
        else:
            # If the wavelength is not in the parent folder name, we need a legend file
            if (path.parent / "legend.docx").is_file():
                legend_file = path.parent / "legend.docx"
            elif (path.parent / "Legend.docx").is_file():
                legend_file = path.parent / "Legend.docx"
            else:
                print(f"No legend for {path}")
                continue

            document = docx.Document(legend_file)
            for paragraph in document.paragraphs:
                # Search for file IDs
                matches = re.findall(r"\d\d[.]\d\d[.]\d\d", paragraph.text)
                if len(matches) == 0:
                    # If no file marker is in the current paragraph, then continue
                    continue
                elif len(matches) > 1:
                    # We should not have more than one file id per paragraph
                    warnings.warn("More than one legend match")
                    print(matches, paragraph.text)
                    #raise ValueError("More than one legend match")
                
                if "400" in paragraph.text and "1000" in paragraph.text:
                    raise ValueError(f"Both 400 and 1000 matches legend file: {legend_file}")
                elif "400" in paragraph.text:
                    wavelength = 400
                elif "1000" in paragraph.text:
                    wavelength = 1000
                else:
                    raise ValueError(f"Unknown wavelength for {legend_file}:\n {paragraph.text}")

                should_break = False
                for match in matches:
                    if match in path.name:
                        should_break = True

                if should_break:
                    break
            else:
                warnings.warn(f"File {path.name} not in legend")
                continue

        # Track particles
        out_path = path.parent / f"{path.stem}.csv"
        failed_path = path.parent / f"{path.stem}_failed"
        start_time = time.time()
        if out_path.is_file() or failed_path.is_file():
            print("Already completed")
            #continue

        try:
            tracks = track_particles(path,)
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
            tracks["Wavelength"] = wavelength
            tracks.to_csv(out_path)
            print(f"Saved tracks: {out_path}")

    print("These files were corrupt or failed:")
    for path, reason in failed.items():
        print(f"Failed {path} as consequence of {reason}")
