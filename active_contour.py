import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import io
from skimage.filters import gaussian
from skimage.segmentation import active_contour


img = io.imread("images/sample.jpg")
img = rgb2gray(img)

s = np.linspace(0, 2*np.pi, 400)
r = 165 + 44*np.sin(s)
c = 167 + 40*np.cos(s)
init = np.array([r, c]).T

snake = active_contour(gaussian(img, 2),
                       init, alpha=0.015, beta=5, gamma=0.001,
                       coordinates='rc')


fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=2)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=2)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()