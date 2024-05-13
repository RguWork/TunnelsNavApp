import cv2
import numpy as np

def translate_image(image, x_offset, y_offset):
    translation_matrix = np.float32([
        [1, 0, x_offset],
        [0, 1, y_offset]
    ])
    result = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return result

def rotate_image(image, theta_degrees):
    height, width = image.shape[0:2]
    center = (image.shape[1]//2, image.shape[0]//2)
    rot_mat = cv2.getRotationMatrix2D(center, theta_degrees, 1)
    
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return result
