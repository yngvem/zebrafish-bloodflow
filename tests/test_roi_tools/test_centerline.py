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


def test_find_all_nearest_centerline_indices_from_roi(rectangular_roi, rectangle_image_shape, rectangle_centerline):
    nearest = centerline_tools.find_all_nearest_centerline_indices_from_roi(
        rectangular_roi, rectangle_image_shape, rectangle_centerline)
    true_nearest = np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1,  0,  1,  2,  3,  4,  5,  6,  7,  8, -1],
        [-1,  0,  1,  2,  3,  4,  5,  6,  7,  8, -1],
        [-1,  0,  1,  2,  3,  4,  5,  6,  7,  8, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ])
    np.testing.assert_allclose(nearest, true_nearest)


def test_find_centerline(rectangular_mask, rectangle_centerline):
    centerline = centerline_tools.find_centerline_from_mask(rectangular_mask)
    assert False