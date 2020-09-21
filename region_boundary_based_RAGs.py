from skimage.future import graph
from skimage import segmentation, color, filters, io
from matplotlib import pyplot as plt


img = io.imread("images/sample.jpg")
gimg = color.rgb2gray(img)

labels = segmentation.slic(img, compactness=25, n_segments=800, start_label=1)
edges = filters.sobel(gimg)
edges_rgb = color.gray2rgb(edges)

g = graph.rag_boundary(labels, edges)
lc = graph.show_rag(labels, g, edges_rgb, img_cmap=None, edge_cmap='viridis',
                    edge_width=1.1)

plt.colorbar(lc, fraction=0.03)
io.show()