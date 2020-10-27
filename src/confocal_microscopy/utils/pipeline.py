from contextlib import contextmanager
import pickle
import h5py
from skimage import io


__all__ = ["Pipeline", "PreviousPipelineValue"]


class PreviousPipelineValue:
    def __init__(self, name):
        self.name = name


class Pipeline:
    def __init__(self, image, name):
        self.name = name
        self.images = {'input': image}
        self.current_name = 'input'
        self.kwargs = {}
        self.functions = {}
        self.call_order = []

    def add_step(self, f, name, kwargs=None, input_name=None, use_checkpoint=True, store_checkpoint=True):
        if input_name is None:
            input_name = self.current_name
        if kwargs is None:
            kwargs = {}
        if name in self.images:
            raise ValueError("Cannot have two steps with same name")

        image = self.images[input_name]

        self.kwargs[name] = kwargs
        self.functions[name] = f
        for key, value in kwargs.items():
            if isinstance(value, PreviousPipelineValue):
                kwargs[key] = self.images[value.name]

        self.call_order.append(name)
        self.current_name = name

        print(f"Performing: {name}")
        with self.open("a") as h5:
            if name in h5 and use_checkpoint:
                self.images[name] = h5[name][:]
                return
    
        self.images[name] = f(image, **kwargs)
        if store_checkpoint:
            with self.open("a") as h5:
                if name in h5:
                    h5[name][:] = self.images[name]
                else:
                    h5[name] = self.images[name]
        return self

    @property
    def current_image(self):
        return self.images[self.current_name]

    def save(self, name):
        with open(f"{name}.pipeline", "wb") as f:
            pickle.dump(f, self)

    @classmethod
    def from_file(self, name):
        with open(f"{name}.pipeline", "rb") as f:
            return pickle.load(f)

    def save_step(self, step, scale=None, dtype=None):
        image = self.images[step]
        if scale:
            image *= scale
        if dtype:
            image = image.astype(dtype)

        io.imsave(f"{step}.tiff", image)

    @contextmanager
    def open(self, mode):
        h5 = h5py.File(self.name, mode)
        yield h5
        h5.close()