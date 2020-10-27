import numpy as np
from tqdm import trange


__all__ = ["anisotropic_diffusion"]


def perona_malik(x, edge_scale):
    return 2*x/(2*(edge_scale**2) + x**2)


def tukey(x, edge_scale):
    x[np.abs(x) > edge_scale] = 0
    return x*((1 - (x/edge_scale)**2)**2)


def _make_num_borders_array(image):
    ndim = image.ndim
    num_borders = 2*ndim*np.ones_like(image)

    for axis in range(ndim):
        front_slices = [slice(None)]*ndim
        front_slices[axis] = 0
        back_slices = [slice(None)]*ndim
        back_slices[axis] = -1

        num_borders[tuple(front_slices)] -= 1
        num_borders[tuple(back_slices)] -= 1
    
    return num_borders


def anisotropic_diffusion(image, edge_scale, step_size, num_steps, return_copy=True, progress=False):
    if return_copy:
        denoised = image.copy()
    else:
        denoised = image
    
    if progress:
        range_ = trange
    else:
        range_ = range
    
    num_borders = _make_num_borders_array(denoised)
    
    ndim = denoised.ndim
    for i in range_(num_steps):
        diffs = np.zeros_like(denoised)

        for axis in range(ndim):
            forward_slices = [slice(None)]*ndim
            forward_slices[axis] = slice(None, -1, None)
            forward_slices = tuple(forward_slices)

            backward_slices = [slice(None)]*ndim
            backward_slices[axis] = slice(1, None, None)
            backward_slices = tuple(backward_slices)

            reversed_slices = [slice(None)]*ndim
            reversed_slices[axis] = slice(None, None, -1)
            reversed_slices = tuple(reversed_slices)

            diffs[forward_slices] += tukey(np.diff(denoised, axis=axis), edge_scale)
            diffs[backward_slices] += tukey(np.diff(denoised[reversed_slices], axis=axis)[reversed_slices], edge_scale)

        denoised += step_size*diffs/num_borders
    
    return denoised