from Snake import Snake
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

points = []
for angle in np.linspace(0, 2*np.pi, 15):
    points.append([65 + (50 * math.cos(angle)), 65 + (40 * math.sin(angle))])
points = np.array(points, dtype=int)
img = cv2.imread("simg/fuzzy-synthetic.jpg", 0)
f, axarr = plt.subplots(2)
xs, ys = zip(*points)
axarr[0].plot(xs, ys)
axarr[1].plot(xs, ys)
axarr[0].imshow(img)
axarr[1].imshow(np.hypot(*np.gradient(img)))
# standard noise
mean = 0.0
std = 1.0
noisy_img = img + np.random.normal(mean, std, img.shape)


s = Snake(noisy_img, points, 1, 0.5555, 10)
s.run(nsize=4)
s.showDif()
