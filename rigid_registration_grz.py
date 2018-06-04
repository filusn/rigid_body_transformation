# Template file for intensity-based rigid registration algorithm.
# Part of laboratory project for course "Techniki Obrazowania Medycznego" at AGH UST, "Mikroelektronika w Technice i Medycynie".
# Released under MIT License (Grzegorz Węgrzyn and Marek Wodzinski).


# This should be really simplified version.

# This implementation should use SSD or CC as similarity metric.
# Scipy implementation of L-BFGS-B should be used (with automatic numerical gradient).
# Single resolution is enough.
# No analytical gradient is necessary. No regularization is necessary.
# No auto stop is necessary (run for given number of iterations).

import numpy as np
#from matplotlib import pyplot as plt
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
	
	def cost_function(params, *args):
		rigid_matrix = generate_rigid_matrix(params[0], params[1], params[2])
		source_after_rigid = rigid_transformation(source, rigid_matrix, spacing)
		if (similarity_metric=='ssd'):
			cost = -ssd(source_after_rigid, target)
		else:
			cost = -cc(source_after_rigid, target)
		return cost
		
	def callback_function(xk):#, optimize.OptimizeResult state):
		#if(echo == True):
		#print("Iteration ", state.nit, ": x_shift = ", xk[0], ", y_shift = ", xk[1], ", z_rotation = ", xk[2])
		print("x_shift = ", xk[0], ", y_shift = ", xk[1], ", z_rotation = ", xk[2])
		if(state.nit == max_iterations):
			return True
		else:
			return False
		
	def callback_jacobian(x, *args):
		step = 0.001
		rad_step = 0.0002
		
		rigid_matrix_x = generate_rigid_matrix(x[0] + step, x[1], x[2])
		rigid_matrix_y = generate_rigid_matrix(x[0], x[1] + step, x[2])
		rigid_matrix_z = generate_rigid_matrix(x[0], x[1], x[2] + rad_step)
		rigid_matrix_x_p = generate_rigid_matrix(x[0] - step, x[1], x[2])
		rigid_matrix_y_p = generate_rigid_matrix(x[0], x[1] - step, x[2])
		rigid_matrix_z_p = generate_rigid_matrix(x[0], x[1], x[2] - rad_step)
		
		image_x = rigid_transformation(source, rigid_matrix_x, spacing)
		image_y = rigid_transformation(source, rigid_matrix_y, spacing)
		image_z = rigid_transformation(source, rigid_matrix_z, spacing)
		image_x_p = rigid_transformation(source, rigid_matrix_x_p, spacing)
		image_y_p = rigid_transformation(source, rigid_matrix_y_p, spacing)
		image_z_p = rigid_transformation(source, rigid_matrix_z_p, spacing)
		
		if (similarity_metric=='ssd'):
			cost = -ssd(source, target)
			cost_x = -ssd(image_x, target)
			cost_y = -ssd(image_y, target)
			cost_z = -ssd(image_z, target)
			cost_x_p = -ssd(image_x_p, target)
			cost_y_p = -ssd(image_y_p, target)
			cost_z_p = -ssd(image_z_p, target)
		else:
			cost = -cc(source, target)
			cost_x = -cc(image_x, target)
			cost_y = -cc(image_y, target)
			cost_z = -cc(image_z, target)
			cost_x_p = -cc(image_x_p, target)
			cost_y_p = -cc(image_y_p, target)
			cost_z_p = -cc(image_z_p, target)
		
		dx = (cost_x - cost_x_p) / (2 * step)
		dy = (cost_y - cost_y_p) / (2 * step)
		dz = (cost_z - cost_z_p) / (2 * rad_step)
		
		return np.array([dx, dy, dz])
	
	
	shift_x = 0
	shift_y = 0
	rot_z = 0
	param = (shift_x, shift_y, rot_z)
	t = optimize.minimize(cost_function, param, method='L-BFGS-B', jac=callback_jacobian, callback=callback_function, options={'disp':echo, 'maxiter':max_iterations})
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
									]	)
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
	#r_spacing = spacing#(1.0, 1.0)
	#rxoom = r_spacing[0] / min(r_spacing)
	#ryoom = r_spacing[1] / min(r_spacing)
	
	x_origin = (image.shape[1]-1) * sxoom / 2
	y_origin = (image.shape[0]-1) * syoom / 2
	
	Cs = np.array(	[
						[1, 0, -x_origin],
						[0, 1, -y_origin],
						[0, 0, 1		]
					]	)
	Cr = np.array(	[
						[1, 0, x_origin	],
						[0, 1, y_origin	],
						[0, 0, 1		]
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
	x_deformation_field = np.reshape(deformation_field[0,:], [image.shape[0], image.shape[1]])# - grid_x
	y_deformation_field = np.reshape(deformation_field[1,:], [image.shape[0], image.shape[1]])# - grid_y
	return x_deformation_field, y_deformation_field


def image_warp(array, u_x, u_y, order=1):
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
	image_warped = ndimage.interpolation.map_coordinates(array, [indices_y, indices_x], order=order)
	image_warped = image_warped	[
									0 : array.shape[0],
									0 : array.shape[1]
								]
	return image_warped


def rigid_transformation(image, matrix, spacing=(1.0, 1.0)):
	"""
	@params image: image to transform rigidly (YxX)
	@params matrix: homogeneous transformation matrix (3x3)
	@params spacing: 2-D tuple with spacing between X and Y pixels respectively
	@return image transformed rigidly (YxX)
	"""
	x_def_field, y_def_field = rigid_dot(image, np.linalg.inv(matrix), spacing)
	image_after_rigid = image_warp(image, x_def_field, y_def_field, order=1)
	return image_after_rigid


def cc(source, target):
	"""
	@params source: source array (YxX)
	@params target: target array (YxX)
	@returns cross correlation between source and target
	Warning: arrays must have the same shape, otherwise ValueError will be thrown
	"""
	if (source.shape == target.shape):
		N = source.shape[0] * source.shape[1]
		result = (1.0/N) * (((source-source.mean())*(target-target.mean()))/(source.std()*target.std())).sum()
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
		return -ssd_result
	else:
		raise ValueError()


