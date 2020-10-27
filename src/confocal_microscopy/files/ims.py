import h5py

def load_image_stack(path, resolution_level=0, time_point=0, channel=0):
    with h5py.File(path, "r") as h5:
        dataset = h5[f"DataSet/"]
        resolution_group = dataset[f"ResolutionLevel {resolution_level}"]
        time_group = resolution_group[f"TimePoint {time_point}"]
        image_group = time_group[f"Channel {channel}"]
        image = image_group["Data"][:]
    
    return image

