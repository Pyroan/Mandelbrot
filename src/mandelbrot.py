# Runs on GPU for that speedy zoom zoom go fast

from numba import jit, vectorize
import numpy as np
from math import sqrt
from PIL import Image, ImageDraw
# from colorsys import hsv_to_rgb
import time
from typing import Tuple
from util import hsv_to_rgb


@jit(nopython=True)
def escape_time(point: Tuple[float, float], max_iterations: int):
    """ Return the escape time for the given worldspace point.
    """
    x, y = point

    # Fast version of mandlebrot iterations
    zx, zy = 0, 0
    a2, b2 = 0, 0
    i = 0
    while a2 + b2 <= 1 << 16 and i < max_iterations-1:
        zy = 2 * zx * zy + y
        zx = a2 - b2 + x
        a2 = zx * zx
        b2 = zy * zy
        i += 1
    return i


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
            i = escape_time((a, b), max_iterations)
            arr[x][y] = i
    return arr


@jit(nopython=True)
def draw_mandelbrot(array, max_iterations):
    """ Return's a Pillow Image coloring the given array of integers,
    Mapping to 
    """
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


def mandelbrot_path(im_size, start_point, xlength, center, max_iterations) -> list:
    """ Returns an ordered list of pixel coordinates showing the results of
        each iteration for the given woorld coordinates starting point
    """
    aspect = im_size[0] / im_size[1]
    ylength = xlength / aspect

    a = start_point[0]
    b = -start_point[1]
    z = complex(a, b)

    points = []

    for _ in range(max_iterations):
        x = (im_size[0] * (z.real - center[0] + (xlength/2))) / xlength
        y = (im_size[1] * (z.imag - center[1] + (ylength/2))) / ylength

        # points.append((-(im_size[0]//2-x), -(im_size[1]//2-y)))
        points.append((x, y))

        if z.real*z.real + z.imag*z.imag > 4:
            break
        z = z * z + complex(a, b)

    return points


def draw_path(im, points):
    point_size = 20
    draw = ImageDraw.Draw(im)
    prev_point = None
    i = 0
    for point in points:
        x, y = point
        v = int((i/len(points) * .5 + .5) * 255)
        draw.ellipse([x-point_size//2, y-point_size//2, x+point_size //
                      2, y+point_size//2], fill=(v, v, v))

        if prev_point != None:
            draw.line([point, prev_point], fill=(v, v, v), width=3)

        prev_point = point
        point_size = 10
        i += 1

    return im
