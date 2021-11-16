"""Discontinued methodology.
"""

from confocal_microscopy.tracking import estimate_piv
import h5py
from pathlib import Path


def find_and_store_profiles(
    data_path,
    n_jobs=4,
    window_size=32,
    overlap=24,
    search_area_size=32,
    skip_existing=True
):
    save_file = data_path.parent / f"{data_path.stem}_piv-velocities-large-window-16overlap.h5"

    velocities__µm_per_s, coords_x, coords_y = estimate_piv.track_particles(
        data_path,
        n_jobs=n_jobs,
        window_size=window_size,
        overlap=overlap,
        search_area_size=search_area_size,
        morphology=True,
    )

    print("Storing data...")
    with h5py.File(save_file, "w") as h5:
        velocities_h5 = h5.create_dataset(
            name="velocities",
            data=velocities__µm_per_s,
            chunks=velocities__µm_per_s.shape,
            compression="gzip",
            compression_opts=7
        )
        velocities_h5.attrs['unit'] = '1e-6 m/s'
        h5.create_dataset(
            name="coords_x",
            data=coords_x,
            chunks=coords_x.shape,
            compression="gzip",
            compression_opts=7
        )
        h5.create_dataset(
            name="coords_y",
            data=coords_y,
            chunks=coords_y.shape,
            compression="gzip",
            compression_opts=7
        )


if __name__ == "__main__":
    parent = Path("/media/yngve/TOSHIBA EXT (YNGVE)/fish_data/organised/7 DAY OLD Fish without tumors/Fish **")
    files = list(parent.glob("**/*red ch_*.ims"))
    failed = []
    
    for i, data_path in enumerate(files):
        print(f"Analysing file {i} out of {len(files)}...")
        try:
            find_and_store_profiles(
                data_path,
                n_jobs=4,
                window_size=32,
                overlap=16,
                search_area_size=32,
                skip_existing=True,
            )
        except Exception as e:
            failed.append(data_path)
            print(f"Failed at {data_path}")
            print(e)
    
    print("Finished, failed at the following files:")
    for f in failed:
        print(f)

