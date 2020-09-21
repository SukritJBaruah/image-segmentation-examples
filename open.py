from skimage import io
from matplotlib import pyplot as plt

img = io.imread("images/sample.jpg")

plt.imshow(img)
plt.show()

