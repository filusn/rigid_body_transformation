import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import io
from skimage import color
from skimage import measure
import time
import sys
# sys.path.append("E:/Python/Rigid/")
import rigid_registration_grz as rr
	
def load_local_image(name):
	current_path = os.path.abspath(os.path.dirname(__file__))
	image_path = os.path.join(current_path, name)
	image = io.imread(image_path)
	return image
def Show(image, title, colors):
	plt.figure()
	plt.title(title)
	plt.imshow(image, cmap=colors)	
image_source = color.rgb2gray(load_local_image("Z6.png"))
image_destination = color.rgb2gray(load_local_image("K1.png"))
#image_source = image_source / np.max(image_source)
image = np.array([[5, 4, 3, 2], [2, 1, 0, 2], [2, 1, 0, 2], [2, 1, 0, 2], [2, 1, 0, 2]])
x = 10
y = 10
z = 0.2#1.76
rigid = rr.generate_rigid_matrix(x, y, z)
print(rigid)
#
# print("Image source:")
# print("Shape: ", image.shape)
#
# t = rr.rigid_dot(image, rigid)
#
# image_after_rigid = rr.rigid_transformation(image_source, np.linalg.inv(rigid), spacing=(1.0,1.0))
# print("Image source:")
# print("Shape: ", image_source.shape)
# print("After rigid:")
# print("Shape: ", image_after_rigid.shape)
#
# translation = rr.rigid_registration(image_source, image_after_rigid, (1.0, 1.0), 20, 'ssd', True)
# print("Calculated transformation: ", translation.x)
# print("Used transformation: ", (x, y, z))
#
# rigid_calc = rr.generate_rigid_matrix(translation.x[0], translation.x[1], translation.x[2])
# image_after_rigid_calc = rr.rigid_transformation(image_source, np.linalg.inv(rigid_calc), spacing=(1.0,1.0))
#
# Show(image_source, "Image source", 'gray')
# Show(image_after_rigid, "Image after rigid", 'gray')
# Show(image_after_rigid_calc, "Image after calc rigid", 'gray')
# plt.show()
#
#
