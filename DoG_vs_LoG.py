import matplotlib.pyplot as plt
from math import sqrt
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.filters import laplace
from skimage.filters import difference_of_gaussians
from skimage.filters import threshold_otsu

image = io.imread("images/synimg.jpg")
#image = io.imread("images/sample.jpg")
image_gray = rgb2gray(image)

fig, axes = plt.subplots(3, 5, figsize=(16, 9))
fig.suptitle('Increase in value of Sigma(σ) --------->')

sigma = 1
#LoG
#gaussian filter with sigma value as second arguement
img_g = gaussian(image_gray, sigma)
#laplacian
img_log = laplace(img_g, ksize=3, mask=None)

#DoG
sigma2 = sigma * sqrt(2)
img_dog = difference_of_gaussians(image_gray, sigma, sigma2)

axes[0, 0].imshow(image)
axes[0, 0].set_title('Image')
axes[1, 0].imshow(img_log)
axes[1, 0].set_title('LoG, σ = 1')
axes[2, 0].imshow(img_dog)
axes[2, 0].set_title('DoG, σ1 = 1, σ2 = √2')

sigma = 0.1
y= 1
#threshold part: edge detection
while sigma < 1.7:
    img_g = gaussian(image_gray, sigma)
    # laplacian
    img_log = laplace(img_g, ksize=3, mask=None)
    # DoG
    sigma2 = sigma * sqrt(2)
    img_dog = difference_of_gaussians(image_gray, sigma, sigma2)

    thresh1 = threshold_otsu(img_log)
    img_log_ed = img_log > thresh1
    thresh2 = threshold_otsu(img_dog)
    img_dog_ed = img_dog > thresh2

    axes[0, y].imshow(img_g)
    axes[0, y].set_title('Gaussian, σ = ' + str(sigma))
    axes[1, y].imshow(img_log_ed, cmap=plt.cm.gray)
    axes[1, y].set_title('LoG edge detection')
    axes[2, y].imshow(img_dog_ed, cmap=plt.cm.gray)
    axes[2, y].set_title('DoG edge detection')

    sigma = sigma + 0.5
    y = y + 1


plt.show()