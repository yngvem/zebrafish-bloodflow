import numpy as np
import scipy.ndimage as ndimage
import networkx

from shapely.geometry import Polygon
from skimage.draw import polygon2mask
from skimage.morphology import skeletonize
from sklearn.neighbors import kneighbors_graph
from numba import njit, prange


@njit(nogil=True, fastmath=True, cache=True)
def _populate_nearest_centerline_idx(centerline_idx, roi_indices_0, roi_indices_1, centerline):
    idx = np.zeros((1, 2))
    for i in range(len(roi_indices_0)):
        idx_0 = roi_indices_0[i]
        idx_1 = roi_indices_1[i]

        idx[0, 0] = idx_0
        idx[0, 1] = idx_1
        centerline_idx[idx_0, idx_1] = np.argmin(
            np.sum((np.array([[idx_0, idx_1]]) - centerline)**2, axis=1)
        )

    return centerline_idx


def find_all_nearest_centerline_indices(mask, centerline):
    """Finds the index of the nearest point on the centerline for all nonzero elements of ``mask``.

    Arguments
    ---------
    mask : np.ndarray(ndim=2)
    centerline : np.ndarray(shape=(N, 2))
        Indices of the centerline
    
    Returns
    -------
    np.ndarray(ndim=2, dtype=int)
        Array with the index of the nearest centerline pixel. Negative for all points outside the ROI.
    """
    roi_indices = np.array(np.nonzero(mask))
    centerline_idx = -np.ones(mask.shape, dtype=np.int32)
    _populate_nearest_centerline_idx(centerline_idx, roi_indices[0], roi_indices[1], centerline)
    return centerline_idx


def find_all_nearest_centerline_indices_from_roi(roi, shape, centerline):
    mask_img = polygon2mask(shape, np.stack((roi['y'], roi['x'])).T)
    return find_all_nearest_centerline_indices(mask_img, centerline)


def make_clipping_polygon(normal, midpoint, bounds):
    """Returns a square with side-length=``2*bounds``

    The square is defined by the ``midpoint`` variable, which is the midpoint of
    one of the sides. The ``normal`` variable contains the normal vector of the
    side that passes through ``midpoint``.

    Arguments
    ---------
    normal : np.array(shape=(2,))
        Normal vector defining the rotation of the square.
        Two sides are normal to the normal-vector and two are parallel
    midpoint : np.array(shape=(2,))
        Midpoint for one of the sides.
    bounds : float
        0.5 * Side length
    
    Returns
    -------
    np.array(shape=(4, 2))
        Array of corners in the square

    Examples
    --------

    >>> make_clipping_polygon([0, 1], [0, 0], 1)

    array([[0, 1], [2, 1], [2, -1], [0, -1]])
    """
    normal, midpoint = np.asarray(normal), np.asarray(midpoint)
    if abs(normal[0]) < 1e-5:
        slope = np.array([1, 0])
    elif abs(normal[1]) < 1e-5:
        slope = np.array([0, 1])
    else:
        slope = np.array([1, -normal[0]/normal[1]])
    
    assert normal.T@slope < 1e-5
    normal = normal / np.linalg.norm(normal)
    slope = slope / np.linalg.norm(slope)
    polygon = [slope, (slope + 2*normal), -slope + 2*normal, -slope, slope]
    return np.stack([p*bounds + midpoint for p in polygon])


def find_line_endpoints(skeleton_img):
    """Find all endpoints in a skeletonised image.

    Returns the indices of all points that only contain one 8-neighbourhood neighbour.
    
    Arguments
    ---------
    skeleton_img : np.array
        n-dimensional skeletonised image
    
    Returns
    -------
    np.ndarray(shape=(P, 2), dtype=int)
        An array of endpoints in the skeletonised image
    """
    connectivity_count_img = ndimage.convolve(skeleton_img.astype(float), np.ones((3, 3)))
    endpoints = np.array(np.nonzero((connectivity_count_img == 2)*skeleton_img)).T
    return endpoints


def find_centerline_coordinates(skeleton_img, start, end, k_neighbours=2):
    """Find the coordinates of the shortest path between ``start`` and ``stop`` in a skeletonised image.
    
    Arguments
    ---------
    skeleton_img : np.array
        n-dimensional skeletonised image
    start_idx : iterable[int]
        Index of start-point. Must be the index of a non-zero element of ``skeleton_img``
    end_idx : iterable[int]
        Index of end-point. Must be the index of a non-zero element of ``skeleton_img``
    k_neighbours : int
        Number of neighbours to use in the neighbourhood graph
    
    Returns
    -------
    np.ndarray(shape=(N, 2), dtype=int)
        An array of centerline coordinates, sorted so neighbouring parts of the
        centerline are neighbouring rows in the array.
    """
    assert skeleton_img[tuple(start)] != 0
    assert skeleton_img[tuple(end)] != 0

    centerline_coords = np.array(np.nonzero(skeleton_img)).T
    start_idx = centerline_coords.tolist().index(list(start))
    end_idx = centerline_coords.tolist().index(list(end))

    knn_graph = networkx.Graph(kneighbors_graph(centerline_coords, k_neighbours))
    path = networkx.shortest_path(knn_graph, start_idx, end_idx)
    return centerline_coords[path, :]


def find_centerline_from_mask(mask, k_neighbours=2):
    """Use Lee's method to skeletonise the image and extract the centerline coordinates.

    Arguments
    ---------
    mask : np.ndarray
        Boolean mask, 1 inside the ROI and 0 outside.

    Returns
    -------
    np.ndarray(shape=(N, 2), dtype=int)
        An array of centerline coordinates, sorted so neighbouring parts of the
        centerline are neighbouring rows in the array.
    """
    # Skeletonize using the method of Lee et al.
    skeleton_img = (skeletonize(mask, method='lee') != 0).astype(float)
    centerline = np.array(np.nonzero(skeleton_img)).T

    # Find endpoints
    endpoints = find_line_endpoints(skeleton_img.astype(float))
    assert len(endpoints[0]) == 2

    # Find shortest path between endpoints
    return find_centerline_coordinates(skeleton_img, endpoints[0], endpoints[1], k_neighbours)


def clip_roi_based_on_centerline(roi, centerline, bounds, normal_estimation_length=2):
    """
    Remove the part of the ROI that extends past the centerline. 

    This removal is done by forming a clipping square defined by a normal vector
    pointing in the same direction of the centerline at the end.

    Arguments
    ---------
    roi : dict[str, list[float]]
        Dictionary containing two vertex lists, one for the x coordinate of each
        vertex and one with the y coordinate of each vertex. These vertices form
        a polygonal ROI.
    centerline : np.ndarray(shape=(N, 2), dtype=int)
        An array containing the centerline coordinates with adjacent coordinates
        being neighbouring rows of the array.
    bounds : float
        Used to define the lengths of the ROI clipping square
    normal_estimation_length : int
        Used to specify how many steps along the centerline we should move to define
        the normal of the clipping square. Too small and it's sensitive to single
        pixel changes in the centerline. Too large and the curvature of the centerline
        affects the normal direction.
    
    Returns
    -------
    dict[str, list[float]]
        The clipped ROI.
    """
    # Find normal vectors
    start_normal = centerline[normal_estimation_length] - centerline[0]
    end_normal = (centerline[-1] - centerline[-(normal_estimation_length + 1)])

    start_clipping_polygon = make_clipping_polygon(-start_normal, centerline[0], bounds)
    end_clipping_polygon = make_clipping_polygon(end_normal, centerline[-1], bounds)
    
    new_shape = Polygon(zip(roi['x'], roi['y']))
    start_halfspace = Polygon(start_clipping_polygon)
    end_halfspace = Polygon(end_clipping_polygon)
    new_shape = new_shape.difference(start_halfspace).difference(end_halfspace)
    print(new_shape)
    #debug_trace()
    return {
        'x': new_shape.exterior.xy[0].tolist(),
        'y': new_shape.exterior.xy[1].tolist()
    }

def find_centerline_and_clip_roi(roi, shape, k_neighbours=2):
    """
    Arguments
    ---------
    roi : dict[str, list[float]]
        Dictionary containing two vertex lists, one for the x coordinate of each
        vertex and one with the y coordinate of each vertex. These vertices form
        a polygonal ROI.
    shape : tuple[int]
        The shape of the full image where the roi is from.
    k_neighbours : int
        Number of neighbours used to generate the KNN graph used for centerline
        ordering.
    """
    mask_img = polygon2mask(shape[::-1], np.stack((roi['x'], roi['y'])).T)
    centerline = find_centerline_from_mask(mask_img, k_neighbours=k_neighbours)
    roi = clip_roi_based_on_centerline(roi, centerline, max(*shape))

    return roi, centerline


def find_centerline_direction(centerline):
    """Finds the direction (normalised gradient) of the centerline.
    
    Uses forward difference on first point, backward difference on last and central on rest.

    Arguments
    ---------
    centerline : np.ndarray(shape=(N, 2), dtype=int)
        An array containing the centerline coordinates with adjacent coordinates
        being neighbouring rows of the array.
    """
    direction = np.zeros_like(centerline)
    direction[0] = centerline[1] - centerline[0]
    direction[-1] = centerline[-1] - centerline[-2]
    direction[1:-1] = (centerline[2:] - centerline[:-2])
    return direction / np.linalg.norm(direction, axis=1)