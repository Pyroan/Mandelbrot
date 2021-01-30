from numba import jit
from typing import Tuple
import numpy as np


@jit(nopython=True)
def hsv_to_rgb(h, s, v):
    """
        Seems like this needs to be @jit for other numba functions to use it
        Which is somewhat frustrating.
    """
    if s == 0.0:
        return np.array([v, v, v])

    i = int(h*6.0)
    f = (h*6.0)-i
    p, q, t = int(255*(v*(1.0-s))), int(255 * (v*(1.0-s*f))
                                        ), int(255*(v*(1.0-s*(1.0-f))))
    v *= 255
    i %= 6
    if i == 0:
        return np.array([v, t, p])
    if i == 1:
        return np.array([q, v, p])
    if i == 2:
        return np.array([p, v, t])
    if i == 3:
        return np.array([p, q, v])
    if i == 4:
        return np.array([t, p, v])
    if i == 5:
        return np.array([v, p, q])


def lerp(a: float, b: float, t: float) -> float:
    return ((1-t) * a) + (t * b)
