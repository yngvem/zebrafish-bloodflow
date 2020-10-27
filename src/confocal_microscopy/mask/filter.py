from scipy import ndimage


def filter_small_regions(mask, min_size, out=None):
    if out is None:
        out = mask.copy()
    elif out is not mask:
        out[:] = mask

    labelled, num_labels = ndimage.label(mask)
    for label in range(num_labels):
        label += 1
        label_mask = (labelled == label)
        if label_mask.sum() < min_size:
            out[label_mask] = 0
    
    return out
