import papermill as pm
from pathlib import Path
from subprocess import run


parent = Path("/media/yngve/TOSHIBA EXT (YNGVE)/fish_data/organised/")
vessels = sorted({p.parent for p in parent.glob("**/*red ch_*.ims")})
for i, image_parent in enumerate(vessels):
    print("Vessel: ", i, "of", len(vessels))

    try:
        bg_path = next(image_parent.parent.glob("*Snap*.ims"))
    except StopIteration:
        try:
            bg_path = next(image_parent.glob("*Snap*.ims"))
        except StopIteration:
            print("No background")
            continue
    

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

