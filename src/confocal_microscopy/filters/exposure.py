import numpy as np


__all__ = ["reduce_dynamic_range", "normalise"]


def reduce_dynamic_range(image, range_min, range_max, out=None):
    """Reduce the dynamic range to lie between the `range_min` percentile and `range_max` percentile.
    """
    min_value = np.percentile(image, range_min)
    max_value = np.percentile(image, range_max)

    if out is None:
        out = image.copy()
        out[image < min_value] = min_value
        out[image > max_value] = max_value
    elif out is image:
        out[image < min_value] = min_value
        out[image > max_value] = max_value
    else:
        out[:] = image
        out[image < min_value] = min_value
        out[image > max_value] = max_value
    
    return out


def normalise(image, out=None):
    if out is None:
        out = image.astype(float) - image.min()
        out /= image.max()
    elif out is image:
        out -= image.min()
        out /= image.max()
    else:
        out[:] = image
        out -= image.min()
        out /= image.max()
    return out

    