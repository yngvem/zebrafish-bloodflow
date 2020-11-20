import h5py

def load_image_stack(path, resolution_level=0, time_point=0, channel=0):
    with h5py.File(path, "r") as h5:
        dataset = h5[f"DataSet/"]
        resolution_group = dataset[f"ResolutionLevel {resolution_level}"]
        time_group = resolution_group[f"TimePoint {time_point}"]
        image_group = time_group[f"Channel {channel}"]
        image = image_group["Data"][:]
    
    return image


def load_physical_image_size(path):
    with h5py.File(path, "r") as h5:
        image_info = h5["DataSetInfo/Image"].attrs
        max_vals = [image_info["ExtMax2"], image_info["ExtMax0"], image_info["ExtMax1"]]
        min_vals = [image_info["ExtMin2"], image_info["ExtMin0"], image_info["ExtMin1"]]

        # Decode byte-string into float
        max_vals = [float(max_val.tobytes().decode("ASCII")) for max_val in max_vals]
        min_vals = [float(min_val.tobytes().decode("ASCII")) for min_val in min_vals]

    return [max_val - min_val for max_val, min_val in zip(max_vals, min_vals)]


def parse_config(file):
    with open(file, "r") as f:
        config_data = f.readlines()
    data = {}
    stack = [data]
    curr_level = 0
    prev_level = 0
    for line in config_data:
        while line.startswith("\t"):
            curr_level += 1
            line = line[1:]
        if curr_level < prev_level:
            num_levels_up = prev_level - curr_level
            for level in num_levels_up:
                stack.pop()
        line = line.strip()
        if line[0] == "[":
            name = line.replace("[", "").replace("]", "")
            data[name] = {}
            stack.append(data[name])
        elif line[0] == "{":
            key = line.split("{DisplayName=")[1].split(", Value")[0]
            value = line.split("Value=")[1][:-1]
        else:
            key, value = line.split("=")
        stack[-1][key] = value
    return data
