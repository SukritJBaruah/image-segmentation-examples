from skimage import io
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

img = io.imread("images/sample.jpg")
img_gray = rgb2gray(img)

plt.imshow(img_gray, cmap=plt.cm.gray)
plt.show()