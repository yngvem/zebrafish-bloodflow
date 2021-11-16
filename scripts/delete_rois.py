from pathlib import Path


fish_path = Path("/media/yngve/TOSHIBA EXT (YNGVE)/fish_data/organised/7 DAY OLD Fish without tumors")
for background_path in (sorted(fish_path.glob("**/*Snap*.ims"))):
    vertex_file = background_path.parent/f"{background_path.stem}_vertices.json"
    if vertex_file.is_file():
        vertex_file.unlink()
