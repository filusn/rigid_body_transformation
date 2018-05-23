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
from skimage import io, color
from scipy.ndimage.interpolation import map_coordinates
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
    @returns transformation parameters (x_translation, y_translatin, z_rotation)
    """

    def cost_function(params, *args):
        if similarity_metric == 'ssd':
            pass
        elif similarity_metric == 'cc':
            pass

    def callback_function(params):
        raise NotImplementedError()

    x_translation = 0
    y_translation = 0
    z_rotation = 0

    optimized = optimize.minimize(cost_function, (x_translation, y_translation, z_rotation),
                                  method='L-BFGS-B', options={'disp': echo, 'maxiter': max_iterations})

    return


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

    matrix = np.matmul(cr_matrix, matrix)
    matrix = np.matmul(matrix, cs_matrix)

    return matrix


def rigid_dot(image, matrix, spacing=(1.0, 1.0)):
    """
    @params image: image with reference size (YxX)
    @params matrix: homogenous transformation matrix (3x3)
    @params spacing: 2-D tuple with spacing between X and Y pixels respectively
    @returns: x deformation field, y_deformation field
    """
    raise NotImplementedError()


def image_warp(array, u_x, u_y, order):
    """
    @params array: array to warp (YxX)
    @params u_x: x deformation field (YxX)
    @params u_y: y deformation field (YxX)
    @params order: interpolation order
    @returns warped image (YxX)
    """

    return map_coordinates(array, [u_x, u_y], order=order)
    # raise NotImplementedError()


def rigid_transformation(image, matrix, spacing=(1.0, 1.0)):
    """
    @params image: image to transform rigidly (YxX)
    @params matrix: homogenous transformation matrix (3x3)
    @params spacing: 2-D tuple with spacing between X and Y pixels respectively
    @params x_origin: optional x origin
    @params y_origin: optional y origin
    """
    raise NotImplementedError()


def cc(source, target):
    """
    @params source: source array (YxX)
    @params target: target array (YxX)
    @returns cross correlation between source and target
    Warning: arrays must have the same shape, otherwise ValueError will be thrown
    """
    raise NotImplementedError()


def ssd(source, target):
    """
    @params source: source array (YxX)
    @params target: target array (YxX)
    @returns sum of squared differences (divided by image size)
    Warning: arrays must have the same shape, otherwise ValueError will be thrown
    """
    raise NotImplementedError()


if __name__ == "__main__":
    source = io.imread(r"C:\Users\Filip\Desktop\TOM_Templates\Templates\Rigid_Template\K1.png")
    # target = io.imread(r"C:\Users\Filip\Desktop\TOM_Templates\Templates\Rigid_Template\K2.png")
    source = color.rgb2gray(source)
    # target = color.rgb2gray(target)
    # plt.figure()
    # plt.imshow(source, cmap='gray')
    # plt.show()

    rigid_matrix = generate_rigid_matrix(0, 0, np.pi / 4)
    transf_matrix = center_matrix(source, rigid_matrix)

    grid_x, grid_y = np.meshgrid(np.arange(source.shape[1]), np.arange(source.shape[0]))

    ones = np.ones((source.shape[0], source.shape[1]))
    vector = np.array([grid_x, grid_y, ones]).reshape(256, 3, 285)
    transfered = np.dot(transf_matrix, vector)
    image = image_warp(source, transfered[0], transfered[1], 1)
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()
