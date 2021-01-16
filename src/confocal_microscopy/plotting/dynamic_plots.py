import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def implay(image_stack, *args, fig=None, ax=None, **kwargs):
    if ax is None:
        fig = plt.gcf()
        ax = plt.gca()

    imshow = ax.imshow(image_stack[0], *args, **kwargs)

    def init():
        return imshow,

    def update(frame):
        i = frame % image_stack.shape[0]
        imshow.set_data(image_stack[i])
        return imshow,


    return FuncAnimation(fig, update, frames=np.arange(0, image_stack.shape[0]),
                         init_func=init, interval=20, blit=True)