# Template file for intensity-based rigid registration algorithm.
# Part of laboratory project for course "Techniki Obrazowania Medycznego" at AGH UST,
# "Mikroelektronika w Technice i Medycynie".
# Released under MIT License (Filip Noworolnik and Marek Wodzinski).


# This should be really simplified version.

# This implementation should use SSD or CC as similarity metric.
# Scipy implementation of L-BFGS-B should be used (with automatic numerical gradient).
# Single resolution is enough.
# No analytical gradient is necessary. No regularization is necessary.
# No auto stop is necessary (run for given number of iterations).
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, color
from scipy.ndimage.interpolation import map_coordinates
from scipy import optimize
import rigid_registration_grz as rr


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

    def cost_function(transf_parameters):
        rigid_matrix = generate_rigid_matrix(transf_parameters[0], transf_parameters[1], transf_parameters[2])
        transformed = rigid_transformation(source, rigid_matrix)
        # show(transformed, 'Transformed', 'gray')
        if similarity_metric == 'ssd':
            cost = ssd(transformed, target)
        elif similarity_metric == 'cc':
            cost = cc(transformed, target)

        return cost

    def callback_function(xk):
        print('Transformation parameters:', xk[0], xk[1], xk[2])

    def jacobian(x, *args):
        step = 0.001
        rad_step = 0.0002

        rigid_matrix_x = generate_rigid_matrix(x[0] + step, x[1], x[2])
        rigid_matrix_y = generate_rigid_matrix(x[0], x[1] + step, x[2])
        rigid_matrix_z = generate_rigid_matrix(x[0], x[1], x[2] + rad_step)
        rigid_matrix_x_p = generate_rigid_matrix(x[0] - step, x[1], x[2])
        rigid_matrix_y_p = generate_rigid_matrix(x[0], x[1] - step, x[2])
        rigid_matrix_z_p = generate_rigid_matrix(x[0], x[1], x[2] - rad_step)

        image_x = rigid_transformation(source, rigid_matrix_x)
        image_y = rigid_transformation(source, rigid_matrix_y)
        image_z = rigid_transformation(source, rigid_matrix_z)
        image_x_p = rigid_transformation(source, rigid_matrix_x_p)
        image_y_p = rigid_transformation(source, rigid_matrix_y_p)
        image_z_p = rigid_transformation(source, rigid_matrix_z_p)

        if similarity_metric == 'ssd':
            cost_x = ssd(image_x, target)
            cost_y = ssd(image_y, target)
            cost_z = ssd(image_z, target)
            cost_x_p = ssd(image_x_p, target)
            cost_y_p = ssd(image_y_p, target)
            cost_z_p = ssd(image_z_p, target)
        else:
            cost_x = cc(image_x, target)
            cost_y = cc(image_y, target)
            cost_z = cc(image_z, target)
            cost_x_p = cc(image_x_p, target)
            cost_y_p = cc(image_y_p, target)
            cost_z_p = cc(image_z_p, target)

        dx = (cost_x - cost_x_p) / (2 * step)
        dy = (cost_y - cost_y_p) / (2 * step)
        dz = (cost_z - cost_z_p) / (2 * rad_step)

        return np.array([dx, dy, dz])

    x_translation = 0
    y_translation = 0
    z_rotation = 0
    optimization = optimize.minimize(cost_function, (x_translation, y_translation, z_rotation),
                                     method='L-BFGS-B', jac=jacobian, callback=callback_function,
                                     options={'disp': echo, 'maxiter': max_iterations})

    return optimization


def generate_rigid_matrix(x_translation, y_translation, z_rotation):
    """
    @params x_translation: translation in x axis (positive values -> right) (in physicial units)
    @params y_translation: translation in y axis (postive values -> down) (in physicial units)
    @params z_rotation: rotation around in radians (positive rotation -> right swirled) (in radians)
    @returns: rigid transformation matrix
    """
    translation_matrix = np.array([[1, 0, x_translation],
                                   [0, 1, y_translation],
                                   [0, 0, 1]])
    rotation_matrix = np.array([[np.cos(z_rotation), np.sin(z_rotation), 0],
                                [-np.sin(z_rotation), np.cos(z_rotation), 0],
                                [0, 0, 1]])

    return np.dot(translation_matrix, rotation_matrix)


def center_matrix(image, matrix, spacing=(1.0, 1.0)):
    """ 
    @params image: image with reference size (YxX)
    @params matrix: original matrix
    @params spacing: 2-D tuple with spacing between X and Y pixels respectively
    @returns: centered rigid transformation matrix
    """

    cs_matrix = np.array([[1, 0, -(image.shape[1] - 1) * spacing[0] / 2],
                          [0, 1, -(image.shape[0] - 1) * spacing[1] / 2],
                          [0, 0, 1]])

    cr_matrix = np.array([[1, 0, (image.shape[1] - 1) * spacing[0] / 2],
                          [0, 1, (image.shape[0] - 1) * spacing[1] / 2],
                          [0, 0, 1]])

    matrix = np.matmul(matrix, cs_matrix)
    matrix = np.matmul(cr_matrix, matrix)

    return matrix


def rigid_dot(image, matrix, spacing=(1.0, 1.0)):
    """
    @params image: image with reference size (YxX)
    @params matrix: homogenous transformation matrix (3x3)
    @params spacing: 2-D tuple with spacing between X and Y pixels respectively
    @returns: x deformation field, y_deformation field
    """
    centered = center_matrix(image, matrix)
    grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    vector = np.vstack((grid_x.flatten(), grid_y.flatten(), np.ones(image.shape[0] * image.shape[1])))
    deformation_field = np.dot(centered, vector)

    return np.reshape(deformation_field[0], [image.shape[0], image.shape[1]]), \
           np.reshape(deformation_field[1], [image.shape[0], image.shape[1]])


def image_warp(array, u_x, u_y, order):
    """
    @params array: array to warp (YxX)
    @params u_x: x deformation field (YxX)
    @params u_y: y deformation field (YxX)
    @params order: interpolation order
    @returns warped image (YxX)
    """

    return map_coordinates(array, [u_y, u_x], order=order)


def rigid_transformation(image, matrix, spacing=(1.0, 1.0)):
    """
    @params image: image to transform rigidly (YxX)
    @params matrix: homogenous transformation matrix (3x3)
    @params spacing: 2-D tuple with spacing between X and Y pixels respectively
    @params x_origin: optional x origin
    @params y_origin: optional y origin
    """
    x_deformation, y_deformation = rigid_dot(image, np.linalg.inv(matrix))

    return image_warp(image, x_deformation, y_deformation, order=1)


def cc(source, target):
    """
    @params source: source array (YxX)
    @params target: target array (YxX)
    @returns cross correlation between source and target
    Warning: arrays must have the same shape, otherwise ValueError will be thrown
    """

    return np.sum(((source - source.mean()) * (target - target.mean())) / (source.std() * target.std())) / source.size


def ssd(source, target):
    """
    @params source: source array (YxX)
    @params target: target array (YxX)
    @returns sum of squared differences (divided by image size)
    Warning: arrays must have the same shape, otherwise ValueError will be thrown
    """

    return np.sum((np.square(source - target))) / source.size


def show(image, title, colors):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=colors)
    plt.show()


if __name__ == "__main__":
    source = io.imread('Z6.png')
    source = color.rgb2gray(source)
    source = (source - source.min()) / (source.max() - source.min())
    rigid_matrix = generate_rigid_matrix(50, 50, 0)
    target = rigid_transformation(source, rigid_matrix)
    a = rigid_registration(source, target, similarity_metric='ssd')
