from functools import wraps

import numpy as np
from tqdm import trange

__all__ = ["apply_slicewise"]


def _get_permutation_index(i, slice_axis):
    if i < slice_axis:
        return i+1
    elif i == slice_axis:
        return 0
    else:
        return i


def apply_slicewise(f, axis=0, progress=False):
    range_ = range
    if progress:
        range_ = trange
    @wraps(f)
    def new_f(image, *args, **kwargs):
        slices = [slice(None)]*image.ndim
        out = [None]*image.shape[axis]
        for i in range_(image.shape[axis]):
            slices[axis] = i
            out[i] = f(image[tuple(slices)], *args, **kwargs)

        permutation = [_get_permutation_index(i, axis) for i in range(out[0].ndim + 1)]
        out = np.stack(out, axis=0).transpose(permutation)
        return out

    return new_f

