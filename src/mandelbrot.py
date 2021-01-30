# Runs on GPU for that speedy zoom zoom go fast

from numba import jit, vectorize
import numpy as np
from math import sqrt
from PIL import Image, ImageDraw
# from colorsys import hsv_to_rgb
import time
from typing import Tuple


@jit(nopython=True)
def mandelbrot(arr: np.ndarray, xlength: int, center: Tuple[float, float], max_iterations: int):
    """ Fill the given numpy array with the escape time of each point within it,
    with the visible x axis defined by xlength, and the center point. 
    """
    dimensions = arr.shape
    aspect = dimensions[0] / dimensions[1]
    ylength = xlength/aspect
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            # Convert from screen to world space
            a = center[0] - (xlength/2) + (x * xlength) / dimensions[0]
            b = center[1] - (ylength/2) + (y * ylength) / dimensions[1]

            # Fast version of mandlebrot iterations
            zx, zy = 0, 0
            a2, b2 = 0, 0
            i = 0
            while a2 + b2 <= 1 << 16 and i < max_iterations-1:
                zy = 2 * zx * zy + b
                zx = a2 - b2 + a
                a2 = zx * zx
                b2 = zy * zy
                i += 1

            arr[x][y] = i
    return arr


@jit(nopython=True)
def hsv_to_rgb(h, s, v):
    """
        Seems like this needs to be @jit to work but i'm not sure why
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


@jit(nopython=True)
def set_color(array, max_iterations):
    width, height = array.shape
    new_array = np.zeros((width, height, 3), np.uint8)
    ipp = np.zeros(max_iterations)
    for x in array:
        for y in x:
            ipp[y] += 1
    total = np.sum(ipp)

    for x in range(width):
        for y in range(height):
            it = array[x][y]
            color = 0.0
            if it == max_iterations-1:
                new_array[x][y] = [0, 0, 0]
            else:
                # new_array[x][y] = [221, 221, 221]
                for z in range(it):
                    color += ipp[z]/total
                color = hsv_to_rgb((color + .3) %
                                   1, 1.0, 1.0)
                new_array[x][y] = color.astype(np.uint8)

    return new_array


def drawpath(im, start_point, xlength, center, max_iterations):
    pointsize = 20

    aspect = im.size[0] / im.size[1]
    ylength = xlength/aspect
    a = start_point[0]
    b = -start_point[1]
    points = []
    draw = ImageDraw.Draw(im)
    z = complex(a, b)
    last_point = None
    for i in range(max_iterations):
        x = (im.size[0] * (z.real - center[0] + (xlength/2))) / xlength
        y = (im.size[1] * (z.imag - center[1] + (ylength/2))) / ylength
        points.append((-(im.size[0]//2-x), -(im.size[1]//2-y)))
        draw.ellipse([x-pointsize//2, y-pointsize//2, x+pointsize //
                      2, y+pointsize//2], fill=(255, 255, 255))
        if last_point:
            draw.line([(x, y), last_point], fill=(255, 255, 255), width=3)
        last_point = (x, y)
        pointsize = 10

        if z.real*z.real + z.imag*z.imag > 4:
            print(f'{i} iterations before escape.')
            break
        z = z*z + complex(a, b)
    return im, points


def lerp(a, b, t):
    return ((1-t) * a) + (t * b)
