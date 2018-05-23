# Template file for intensity-based rigid registration algorithm.
# Part of laboratory project for course "Techniki Obrazowania Medycznego" at AGH UST, "Mikroelektronika w Technice i Medycynie".
# Released under MIT License (__________ and Marek Wodzinski).


# This should be really simplified version.

# This implementation should use SSD or CC as similarity metric.
# Scipy implementation of L-BFGS-B should be used (with automatic numerical gradient).
# Single resolution is enough.
# No analytical gradient is necessary. No regularization is necessary.
# No auto stop is necessary (run for given number of iterations).

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import optimize

def rigid_registration(source, target, spacing=(1.0, 1.0), max_iterations=20, similarity_metric='ssd', echo=True):
	"""
	Perform intensity-based rigid registration.
	@params source: source image (YxX)
	@params target: target image (YxX)
	@params spacing: spacing between pixels (x and y respectively)
	@params max_iterations: maximum number of optimizer iterations
	@params similarity_metric: similarity metric to use
	@params echo: print results?
	@returns transformation parameters (x_translation, y_translation, z_rotation)
	"""
	
	def cost_function(array1, array2):
		if (similarity_metric=='ssd'):
			cost = ssd(array1, array2)
		else:
			cost = cc(array1, array2)
		#print("Cost: ", cost)
		return cost
		
	def callback_function(params, *args):
		rigid_matrix = generate_rigid_matrix(params[0], params[1], params[2])
		#print("Params: ", params)
		source_after_rigid = rigid_transformation(source, np.linalg.inv(rigid_matrix), spacing)
		return cost_function(source_after_rigid, target)
		
	callback_f = callback_function
	shift_x, shift_y, rot_z = 0,0,0
	param = (shift_x, shift_y, rot_z)
	t = optimize.minimize(callback_f, param, method='L-BFGS-B',
																options={	'disp':echo, 
																			#'maxcor': 5, 
																			#'ftol': 0,#2.220446049250313e-09, 
																			#'gtol': 0,#1e-05, 
																			#'eps': 1, 
																			#'maxfun': 250, 
																			'maxiter':max_iterations})
	return t

def generate_rigid_matrix(x_translation, y_translation, z_rotation):
	"""
	@params x_translation: translation in x axis (positive values -> right) (in physical units)
	@params y_translation: translation in y axis (positive values -> down) (in physical units)
	@params z_rotation: rotation around in radians (positive rotation -> right swirled) (in radians)
	@returns: rigid transformation matrix
	"""
	rotate_matrix = np.array(	[
									[np.cos(z_rotation)	,-np.sin(z_rotation),0],
									[np.sin(z_rotation)	,np.cos(z_rotation)	,0],
									[0					,0					,1]
								]	)
	translation_matrix = np.array(	[
										[1, 0, x_translation],
										[0, 1, y_translation],
										[0, 0, 1			]
									] 	)
	transformation_matrix = np.dot(translation_matrix, rotate_matrix)
	return transformation_matrix

def center_matrix(image, matrix, spacing=(1.0, 1.0)):
	""" 
	@params image: image with reference size (YxX)
	@params matrix: original matrix
	@params spacing: 2-D tuple with spacing between X and Y pixels respectively
	@returns: centered rigid transformation matrix
	"""
	sxoom = spacing[0] / min(spacing)
	syoom = spacing[1] / min(spacing)
	r_spacing = spacing#(1.0, 1.0)
	rxoom = r_spacing[0] / min(r_spacing)
	ryoom = r_spacing[1] / min(r_spacing)
	
	Cs = np.array(	[
						[1, 0, -(image.shape[1]-1) * sxoom / 2	],
						[0, 1, -(image.shape[0]-1) * syoom / 2	],
						[0, 0, 1								]
					]	)
	Cr = np.array(	[
						[1, 0, ((image.shape[1]-1) * rxoom / 2)	],
						[0, 1, ((image.shape[0]-1) * ryoom / 2)	],
						[0, 0, 1								]
					]	)
	T_R_Cs = np.dot(matrix, Cs)
	Cr_T_R_Cs = np.dot(Cr, T_R_Cs)
	return Cr_T_R_Cs


def rigid_dot(image, matrix, spacing=(1.0, 1.0)):
	"""
	@params image: image with reference size (YxX)
	@params matrix: homogeneous transformation matrix (3x3)
	@params spacing: 2-D tuple with spacing between X and Y pixels respectively
	@returns: x_deformation_field, y_deformation_field
	"""
	centered_matrix = center_matrix(image, matrix, spacing)
	grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
	points = np.vstack((grid_x.flatten(), grid_y.flatten(), np.ones((image.shape[0] * image.shape[1]), dtype=int)))
	deformation_field = np.dot(centered_matrix, points)
	x_deformation_field = np.reshape(deformation_field[0,:], [image.shape[0], image.shape[1]]) - grid_x
	y_deformation_field = np.reshape(deformation_field[1,:], [image.shape[0], image.shape[1]]) - grid_y
	return x_deformation_field, y_deformation_field


def image_warp(array, u_x, u_y, order='test'):
	"""
	@params array: array to warp (YxX)
	@params u_x: x deformation field (YxX)
	@params u_y: y deformation field (YxX)
	@params order: interpolation order
	@returns warped image (YxX)
	"""
	grid_x, grid_y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
	indices_x = u_x + grid_x
	indices_y = u_y + grid_y
	image_warped = ndimage.interpolation.map_coordinates(array, [indices_y, indices_x])
	image_warped = image_warped[
									0 : array.shape[0],
									0 : array.shape[1]
								]
	return image_warped


def rigid_transformation(image, matrix, spacing=(1.0, 1.0), x_origin=0, y_origin=0):
	"""
	@params image: image to transform rigidly (YxX)
	@params matrix: homogeneous transformation matrix (3x3)
	@params spacing: 2-D tuple with spacing between X and Y pixels respectively
	@params x_origin: optional x origin
	@params y_origin: optional y origin
	@return image transformed rigidly (YxX)
	"""
	x_def_field, y_def_field = rigid_dot(image, matrix, spacing)
	image_after_rigid = image_warp(image, x_def_field, y_def_field, order='test')
	return image_after_rigid


def cc(source, target):
	"""
	@params source: source array (YxX)
	@params target: target array (YxX)
	@returns cross correlation between source and target
	Warning: arrays must have the same shape, otherwise ValueError will be thrown
	"""
	if (source.shape == target.shape):
		N = np.ma.size(source)
		uA = source.mean()
		uB = target.mean()
		dA = source.std()
		dB = target.std()
		result = (1.0/N) * (((source-uA)*(target-uB))/(dA*dB)).sum()
		return result
	else:
		raise ValueError()

def ssd(source, target):
	"""
	@params source: source array (YxX)
	@params target: target array (YxX)
	@returns sum of squared differences (divided by image size)
	Warning: arrays must have the same shape, otherwise ValueError will be thrown
	"""
	if (source.shape == target.shape):
		N = (source.shape[0] * source.shape[1])
		diff = np.square(source - target)
		ssd_result = (1.0 / N) * np.sum(diff)
		return ssd_result
	else:
		raise ValueError()


