import os
import random

import cv2
import einops
import numpy as np
import torch
from PIL import Image


def create_canvas_and_transform_image(img_rgba, canvas_size, scale_factor, rotation_angle, translation_xy):
    """
        
    """
    H, W = canvas_size
    img_h, img_w = img_rgba.shape[:2]
    # Step 1: Create a canvas
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    # Step 2: Place img_rgba in the center of the canvas
    start_y, start_x = (H - img_h) // 2, (W - img_w) // 2
    canvas[start_y:start_y+img_h, start_x:start_x+img_w, :] = img_rgba
    # Step 3 & 4: Perform the geometric transformations and superimpose the image
    # Calculate the center of the original image on the canvas
    center = (start_x + img_w // 2, start_y + img_h // 2)
    # Scaling and rotation
    M = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
    # Translation
    M[:, 2] += translation_xy  # Adding the translation values to the transformation matrix
    # Apply affine transformation
    transformed_img = cv2.warpAffine(canvas, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
    # Step 5: Generate a mask from the alpha channel
    mask = transformed_img[:, :, 3] > 0
    return transformed_img, mask


def add_Gaussian_noise(img_add, mean=0,std=25):
    H, W = img_add.shape[:2]
    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, std, (H, W, 3)).astype(np.uint8)
    # Identify transparent areas in the image
    # The alpha channel is the 4th channel in 'img_add'; transparency implies alpha values are 0
    transparent_areas = img_add[:, :, 3] == 0
    # Prepare an empty canvas for the noise to be added
    noise_to_add = np.zeros_like(img_add[:, :, :3])
    # Fill the transparent areas in 'noise_to_add' with the Gaussian noise
    noise_to_add[transparent_areas] = gaussian_noise[transparent_areas]
    # Separate alpha channel
    # alpha_channel = img_add[:, :, 3].copy()
    # Converting 'noise_to_add' to the same dtype as 'img_add' to avoid data type issues during addition
    noise_to_add = noise_to_add.astype(img_add.dtype)

    # Combining noise with the original image
    # First, combine RGB channels while avoiding overflow beyond 255
    final_img = cv2.add(img_add[:, :, :3], noise_to_add)
    # final_img[final_img > 255] = 255
    return final_img


def add_pure_background(img_add):
    # Assuming or creating a mask; for demonstration, let's extract it from the alpha channel
    mask = img_add[:, :, 3] > 0  # This creates a mask where True indicates opaque pixels
    # Separate the RGB channels from 'img_add'
    rgb_img = img_add[:, :, :3]
    # Using the mask to select non-transparent (opaque) pixels in the RGB channels
    non_transparent_pixels = rgb_img[mask]
    # Calculate the average values across all non-transparent pixels for each channel (R, G, B)
    average_rgb_values = np.mean(non_transparent_pixels, axis=0)
    # Prepare a fill image with the same shape as 'rgb_img', filled with 'average_rgb_values'
    fill_img = np.ones_like(rgb_img) * average_rgb_values.astype(int)
    # Use the inverted mask to replace the transparent areas in 'rgb_img' with 'average_rgb_values'
    rgb_img[~mask] = fill_img[~mask]
    return rgb_img.astype(img_add.dtype)

    
base_path = 'figs/outputs'
output_path = 'figs/geo_outputs'
image_path = f'{base_path}/motor_vit_s_pts_mask_BGRA_cropped.png'
mask_path = f'{base_path}/motor_vit_s_pts_mask_cropped.npy'

img_bgra = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # H, W, 4
mask = np.load(mask_path) # H, W

# Example usage
canvas_size = (640, 1024)  # Example canvas size
scale_factor = 0.7  # Example scaling factor
rotation_angle = 15  # Example rotation angle in degrees
translation_xy = (50, -30)  # Example translation (x, y)

# Perform transformations and get the final image and mask
img_add, mask = create_canvas_and_transform_image(img_bgra, canvas_size, scale_factor, rotation_angle, translation_xy)
img_background=add_pure_background(img_add)

cv2.imwrite(os.path.join(output_path, 'motor_mask.png'), mask.astype(np.uint8) * 255)
cv2.imwrite(os.path.join(output_path, 'motor_canvas.png'), img_add)
cv2.imwrite(os.path.join(output_path, 'motor_background.png'), img_background)