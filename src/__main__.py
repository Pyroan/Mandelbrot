import mandelbrot
import numpy as np
import time
from PIL import Image
from util import lerp

width, height = 1920, 1080
dimensions = (width, height)

aspect = dimensions[0] / dimensions[1]
# ylength = xlength / aspect

array = np.zeros(dimensions, int)

# xlength = 4

# p = (-.5, 0)

iterations = 300


# p = (.366363, -.59153375)
p = (-.5, 0)

start = time.time()
i = mandelbrot.mandelbrot(array, 4, p, iterations)
end = time.time()
# print("Elapsed = %s" % (end - start))

start = time.time()
i = mandelbrot.draw_mandelbrot(i, iterations)
end = time.time()
# print("Elapsed (color) = %s" % (end - start))
# print(i)

img = Image.fromarray(i).transpose(Image.ROTATE_90)
# pts = mandelbrot.mandelbrot_path(img.size, (-1.5, 0.1), 4, p, iterations)
# pts = mandelbrot.mandelbrot_path(img.size, (.365, .592), 4, p, iterations)
pts = mandelbrot.mandelbrot_path(img.size, (-1.15, .20), 4, p, iterations)
img = mandelbrot.draw_path(img, pts)
img.save('../output/mainmandelbrot.png')

# frame = 0
# max_frames = 300
# lengths = []
# for i in range(1, len(pts)):
#     lengths.append(sqrt((pts[i][0]-pts[i-1][0]) **
#                         2 + (pts[i][1]-pts[i-1][1])**2))
# full_length = sum(lengths)
# print(f'\t{frame}\t{pts[0][0]}\t{pts[0][1]}')
# for i in range(1, len(pts)):
#     frame += int((lengths[i-1]/full_length)*max_frames) or 1
#     print(f'\t{frame}\t{pts[i][0]}\t{pts[i][1]}')


# function for x length is (i think something like)
# 4 * (10**(-deepest_zoom*frame_number/total_frames))


# frames = 300

# for i in range(frames):
#     completion = i / frames
#     xlength = 4 * (10**(-6*completion))
#     a = mandelbrot(
#         array, xlength, (lerp(-.5, p[0], 1-10**(-6*completion)),
#                          lerp(0, p[1], 1-10**(-6*completion))))
#     a = set_color(a)
#     img = Image.fromarray(a).transpose(Image.ROTATE_90)
#     img.save(f'deepzoom2/mandelbrot_{i}.png')
#     print(f'{i+1}/{frames} complete')
