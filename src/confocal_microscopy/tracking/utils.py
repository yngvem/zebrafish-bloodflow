from ..files import ims


def load_background(path):
    background = ims.load_image_stack(path).squeeze()
    metadata_path = path.parent / f"{path.stem}_metadata.txt"
    metadata = ims.parse_config(metadata_path)

    width = int(metadata["Width"])
    height = int(metadata["Height"])

    return background[:height, :width]