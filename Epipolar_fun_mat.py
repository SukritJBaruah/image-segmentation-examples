import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import matplotlib.pyplot as plt

np.random.seed(0)

#img_left = io.imread("images/library1.jpg")
#img_right = io.imread("images/library2.jpg")
img_left = io.imread("images/Leuven1A.jpg")
img_right = io.imread("images/Leuven1B.jpg")
img_left, img_right = map(rgb2gray, (img_left, img_right))

# Find sparse feature correspondences between left and right image.

descriptor_extractor = ORB()

descriptor_extractor.detect_and_extract(img_left)
keypoints_left = descriptor_extractor.keypoints
descriptors_left = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img_right)
keypoints_right = descriptor_extractor.keypoints
descriptors_right = descriptor_extractor.descriptors

matches = match_descriptors(descriptors_left, descriptors_right,
                            cross_check=True)

# Estimate the epipolar geometry between the left and right image.

model, inliers = ransac((keypoints_left[matches[:, 0]],
                         keypoints_right[matches[:, 1]]),
                        FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=1, max_trials=5000)

inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

print(f"Number of matches: {matches.shape[0]}")
print(f"Number of inliers: {inliers.sum()}")

# Visualize the results.

fig, ax = plt.subplots()

plt.gray()

plot_matches(ax, img_left, img_right, keypoints_left, keypoints_right,
             matches[inliers], only_matches=True)
ax.axis("off")
ax.set_title("Matched Keypoints Inliers")


plt.show()