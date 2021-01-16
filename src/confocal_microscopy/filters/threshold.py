from skimage import filters

__all__ = ["threshold_local"]


def threshold_local(
    image, block_size, method="gaussian", offset=0, mode="reflect", param=None, cval=0
    ):
    th = filters.threshold_local(image, block_size, method=method, offset=offset, mode=mode, param=param, cval=cval)
    return (image > th).astype(float)