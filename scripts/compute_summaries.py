import papermill as pm
from pathlib import Path
from subprocess import run
import multiprocessing
from functools import partial


def run_summary(params, N):
    i, image_parent = params
    print("Vessel: ", i, "of", N)

    try:
        bg_path = next(image_parent.parent.glob("*Snap*.ims"))
    except StopIteration:
        try:
            bg_path = next(image_parent.glob("*Snap*.ims"))
        except StopIteration:
            print("No background")
            return
    

    filename = image_parent / f"summary.ipynb"

    print("Filename: ", filename)
    try:
        pm.execute_notebook(
        'Summary analysis.ipynb',
        filename,
        parameters = dict(image_parent=str(image_parent), bg_path=str(bg_path))
        )
        run(["jupyter", "nbconvert", f"{filename}", "--to=pdf", "--no-input"])

    except Exception as e:
        print(e)
        pass

parent = Path("/media/yngve/TOSHIBA EXT (YNGVE)/fish_data/organised/")
vessels = sorted({p.parent for p in parent.glob("**/*red ch_*.ims")})
for data in enumerate(vessels):
    run_summary(data, N=len(vessels))

#with multiprocessing.Pool(4) as p
#    p.map(partial(run_summary, N=len(vessels)), enumerate(vessels))
