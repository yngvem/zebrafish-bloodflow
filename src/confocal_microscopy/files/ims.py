from abc import ABC, abstractmethod

from tqdm import trange
import numpy as np
import h5py


def load_image_stack(path, resolution_level=0, time_point=0, channel=0):
    with h5py.File(path, "r") as h5:
        dataset = h5[f"DataSet/"]
        resolution_group = dataset[f"ResolutionLevel {resolution_level}"]
        time_group = resolution_group[f"TimePoint {time_point}"]
        image_group = time_group[f"Channel {channel}"]
        image = image_group["Data"][:]
    
    return image


def load_video_stack(path, resolution_level=0, channel=0, progress=False, num_timesteps=None):
    metadata_path = path.parent / f"{path.stem}_metadata.txt"
    metadata = parse_config(metadata_path)

    width = int(metadata["Width"])
    height = int(metadata["Height"])

    if progress:
        range_ = trange
    else:
        range_ = range
    with h5py.File(path, "r") as h5:
        dataset = h5[f"DataSet/"]
        resolution_group = dataset[f"ResolutionLevel {resolution_level}"]
        if num_timesteps is None:
            num_timesteps = len(resolution_group)
        dataset_pattern = f"TimePoint {{time_point}}/Channel {channel}/Data"

        frames = [
            resolution_group[dataset_pattern.format(time_point=time_point)][:, :height, :width]
            for time_point in range_(num_timesteps)
        ]
    return np.stack(frames, axis=0)


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
            key, *value = line.split("=")
            value = "=".join(value)
        stack[-1][key] = value
    return data


class LazyIMSVideoLoader(ABC):
    def __init__(
        self,
        path,
        num_timesteps=None,
        resolution_level=0,
        channel=0,
        limits=None,
        progress=True
    ):
        self.path = path
        metadata_path = path.parent / f"{path.stem}_metadata.txt"
        metadata = parse_config(metadata_path)

        self._width = int(metadata["Width"])
        self._height = int(metadata["Height"])
        self._num_timesteps = num_timesteps
        self._resolution_level = resolution_level
        self._dataset_pattern = f"TimePoint {{time_point}}/Channel {channel}/Data"
        self._current_timepoint = 0
        self._limits = limits

        self.should_preprocess = True

        if progress:
            self._range = trange
        else:
            self._range = range

    def __enter__(self):
        self.h5 = h5py.File(self.path, "r")
        try:
            dataset = self.h5["DataSet/"]
            self._resolution_group = dataset[f"ResolutionLevel {self._resolution_level}"]
            if self._num_timesteps is None:
                self._num_timesteps = len(self._resolution_group)

            should_preprocess = self.should_preprocess
            self.should_preprocess = False
            self.background_signal = 0
            print("Computing mean")
            for frame in self:
                self.background_signal += frame.astype(float)
            if self._limits is None:
                print("Computing signal limits")
                self._limits = [np.inf, -np.inf]
                for frame in self:
                    self._limits[1] = max(self._limits[1], np.max(frame - self.background_signal))
                    self._limits[0] = min(self._limits[0], np.min(frame - self.background_signal))
                self._limits[0] = max(0, self._limits[0])
                self._limits = tuple(self._limits)
            self.background_signal /= len(self)
            self.should_preprocess = should_preprocess
        except Exception as e:
            self.h5.close()
            raise e

        return self

    def __iter__(self):
        self._time_point_iterator = iter(self._range(self._num_timesteps))
        return self

    def __len__(self):
        return self._num_timesteps

    def __next__(self):
        time_point = next(self._time_point_iterator)
        dataset_name = self._dataset_pattern.format(time_point=time_point)
        self._current_timepoint += 1
        frame = self._resolution_group[dataset_name][:, :self._height, :self._width]
        return self.preprocess(frame.squeeze())

    def __exit__(self, type, value, traceback):
        self.h5.close()

    def __getitem__(self, time_point):
        dataset_name = self._dataset_pattern.format(time_point=time_point)
        frame = self._resolution_group[dataset_name][:, :self._height, :self._width]
        return self.preprocess(frame.squeeze())

    def preprocess(self, frame):
        if not self.should_preprocess:
            return frame
        return self._preprocess(frame)

    @abstractmethod
    def _preprocess(self, frame):
        return frame
