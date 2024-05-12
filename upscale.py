import os
import cv2
import numpy as np
import random

TOLERANCE = 10
MAX_ANGLE_DEGREES = 6
MAX_TRANSLATION_PIXELS = 8

def translate_image(image, x_offset, y_offset):
    translation_matrix = np.float32([
        [1, 0, x_offset],
        [0, 1, y_offset]
    ])
    result = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return result

def rotate_image(image, theta_degrees):
    height, width = image.shape[0:2]
    center = (image.shape[1]//2, image.shape[0]//2)
    rot_mat = cv2.getRotationMatrix2D(center, theta_degrees, 1)
    
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return result

def upscale(directory_path, target_size):
    dir_size = len(list(os.scandir(directory_path)))
    if dir_size == 0:
        return
    if dir_size + TOLERANCE >= target_size:
        return
    repetitions = target_size // dir_size
    remainder = (target_size % dir_size)

    indexes_to_duplicate_extra = set()
    if remainder != 0:
        offset_scale = dir_size / remainder
        for i in range(remainder):
            indexes_to_duplicate_extra.add(int(i * offset_scale))

    with os.scandir(directory_path) as files:
        for index, file in enumerate(files):
            video = cv2.imread(file.path)
            
            # Assume 45 images in folder and want to upscale to 150. 
            # Each image should get two additional copies of itself: (150 // 45) - 1 and preserve the original image
            # This only produces 135 images so need to generate one more every (150 // 15) frames
            extra_frame = index in indexes_to_duplicate_extra
            image_copy_count = repetitions + extra_frame - 1
            for upscale_index in range(image_copy_count):
                file_name, extension = os.path.splitext(file.name)

                theta_degrees = random.randint(-MAX_ANGLE_DEGREES, MAX_ANGLE_DEGREES)
                x_offset = random.randint(-MAX_TRANSLATION_PIXELS, MAX_TRANSLATION_PIXELS)
                y_offset = random.randint(-MAX_TRANSLATION_PIXELS, MAX_TRANSLATION_PIXELS)
                transformed_image = rotate_image(video, theta_degrees)
                transformed_image = translate_image(transformed_image, x_offset, y_offset)
                save_path = os.path.join(directory_path, file_name + f"-{upscale_index + 2}" + extension)
                cv2.imwrite(save_path, transformed_image)
    dir_size = len(list(os.scandir(directory_path)))
