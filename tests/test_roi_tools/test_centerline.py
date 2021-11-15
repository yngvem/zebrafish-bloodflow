import confocal_microscopy.roi_tools.centerline as centerline_tools
import numpy as np
from skimage.draw import polygon2mask
import pytest


@pytest.fixture()
def rectangular_mask():
    return np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

@pytest.fixture()
def rectangular_roi():
    return {'x': [1, 10, 10, 1], 'y': [1, 1, 4, 4]}

@pytest.fixture()
def rectangle_image_shape():
    return 5, 11

@pytest.fixture()
def rectangle_centerline(rectangular_mask):
    return np.array([
        [3, 1+i] for i in range(rectangular_mask.shape[1]-2)
    ])


def test_find_all_nearest_centerline_indices(rectangular_mask, rectangle_centerline):
    nearest = centerline_tools.find_all_nearest_centerline_indices(
        rectangular_mask, rectangle_centerline
    )    

    true_nearest = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  1,  2,  3,  4,  5,  6,  7,  8, -1],
        [-1,  0,  1,  2,  3,  4,  5,  6,  7,  8, -1],
        [-1,  0,  1,  2,  3,  4,  5,  6,  7,  8, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ])

    np.testing.assert_allclose(nearest, true_nearest)


def test_nearest_centerline_direction(rectangular_mask, rectangle_centerline):
    nearest = centerline_tools.find_all_nearest_centerline_indices(
        rectangular_mask, rectangle_centerline
    )
    NaN = np.nan
    true_nearest_y_direction = np.array([
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        [NaN,   1,   1,   1,   1,   1,   1,   1,   1,   1, NaN],
        [NaN,   1,   1,   1,   1,   1,   1,   1,   1,   1, NaN],
        [NaN,   1,   1,   1,   1,   1,   1,   1,   1,   1, NaN],
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
    ])

    true_nearest_x_direction = np.array([
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        [NaN,   0,   0,   0,   0,   0,   0,   0,   0,   0, NaN],
        [NaN,   0,   0,   0,   0,   0,   0,   0,   0,   0, NaN],
        [NaN,   0,   0,   0,   0,   0,   0,   0,   0,   0, NaN],
        [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
    ])

    nearest_direction = centerline_tools.find_nearest_centerline_direction(nearest, rectangle_centerline)
    true_nearest_direction = np.stack([true_nearest_x_direction, true_nearest_y_direction], axis=-1)

    np.testing.assert_allclose(nearest_direction, true_nearest_direction, equal_nan=True)



def test_find_centerline_and_clip_roi_runs(rectangular_roi, rectangle_image_shape, rectangle_centerline):
    centerline_tools.find_centerline_and_clip_roi(rectangular_roi, rectangle_image_shape) 
