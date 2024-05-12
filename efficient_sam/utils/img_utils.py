import cv2
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    
def show_points(coords, labels, ax, marker_size=75):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)   
    
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    

def save_transparent_img(sample_image_np, mask, save_path):
    # 保存为带有透明度的四通道BGRA格式：
    masked_image_np = sample_image_np.copy().astype(np.uint8)
    masked_image_np = cv2.cvtColor(masked_image_np, cv2.COLOR_RGB2BGRA)  # 转换为带有透明度的图像
    # mask = np.logical_not(mask)
    mask = np.where(mask, 255, 0).astype(np.uint8)
    masked_image_np[:,:,3] = mask  # 设置透明度
    # 保存结果为PNG格式
    cv2.imwrite(save_path, masked_image_np)


def save_crop_image_mask(sample_image_np, mask, save_path):
    # 保存为带有透明度的四通道BGRA格式：
    sample_image_np = sample_image_np.copy().astype(np.uint8)
    sample_image_np = cv2.cvtColor(sample_image_np, cv2.COLOR_RGB2BGRA)  # 转换为带有透明度的图像
    mask = np.where(mask, 255, 0).astype(np.uint8)
    sample_image_np[:,:,3] = mask  # 设置透明度
    # 找到蒙版的边界
    top, bottom, left, right = find_border(mask)
    # 根据边界裁剪图像
    cropped_image = sample_image_np[top:bottom, left:right]
    # 保存裁剪后的结果为带有透明度的PNG格式
    cv2.imwrite(save_path, cropped_image)


def find_border(mask):
    top = np.argmax(mask.sum(axis=1) > 0)   # 上边界
    bottom = len(mask) - np.argmax(mask[::-1].sum(axis=1) > 0)# 下边界
    left = np.argmax(mask.sum(axis=0) > 0)  # 左边界
    right = len(mask[0]) - np.argmax(mask[:,::-1].sum(axis=0) > 0)  # 右边界

    return top, bottom, left, right