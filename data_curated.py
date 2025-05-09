from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import torchvision.transforms.transforms as T
from AFS_dataset import RemoveBlackBorders
from PIL import Image
import os
import pandas as pd
import torch
import numpy as np
import cv2

def torch_to_cv2_image(img_tensor):
    """ Torch tensor (C,H,W) → numpy (H,W) with uint8, grayscale """
    img = img_tensor.clone().detach().cpu()
    if img.shape[0] == 3:  # RGB
        img = torch.mean(img, dim=0)  # convert to grayscale
    img = img.numpy()
    img = (img * 255).astype(np.uint8)
    return img

def extract_edge_border(img_np, border_width=50):
    """Extract edge only on border area using Canny"""
    edge = cv2.Canny(img_np, 0, 50)
    h, w = edge.shape
    mask = np.zeros_like(edge, dtype=np.uint8)
    mask[:border_width, :] = 1
    mask[-border_width:, :] = 1
    mask[:, :border_width] = 1
    mask[:, -border_width:] = 1
    return edge * mask  # keep border edges only

def keep_first_edge_per_direction(edge_map):
    h, w = edge_map.shape
    result = np.zeros_like(edge_map)

    # 위→아래
    for col in range(w):
        for row in range(h):
            if edge_map[row, col]:
                result[row, col] = 255
                break

    # 아래→위
    for col in range(w):
        for row in reversed(range(h)):
            if edge_map[row, col]:
                result[row, col] = 255
                break

    # 왼→오
    for row in range(h):
        for col in range(w):
            if edge_map[row, col]:
                result[row, col] = 255
                break

    # 오→왼
    for row in range(h):
        for col in reversed(range(w)):
            if edge_map[row, col]:
                result[row, col] = 255
                break

    return result

def soft_iou_score(mask1, mask2, blur_ksize=11, blur_sigma=1):
    mask1 = cv2.GaussianBlur(mask1.astype(np.float32), (blur_ksize, blur_ksize), blur_sigma)
    mask2 = cv2.GaussianBlur(mask2.astype(np.float32), (blur_ksize, blur_ksize), blur_sigma)

    intersection = np.sum(np.minimum(mask1, mask2))
    union = np.sum(np.maximum(mask1, mask2))
    return intersection / union if union > 0 else 0

def compute_edge_border_iou(tensor1, tensor2, border_width=50):
    img1 = torch_to_cv2_image(tensor1)
    img2 = torch_to_cv2_image(tensor2)

    edge1 = keep_first_edge_per_direction(extract_edge_border(img1, border_width))
    edge2 = keep_first_edge_per_direction(extract_edge_border(img2, border_width))

    return soft_iou_score(edge1, edge2)

transform = T.Compose([RemoveBlackBorders(),T.Resize((224,224)), T.ToTensor()])
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

path = './dataset/test/'
category = os.listdir(path)
rgb_path = path + 'rgb/'
ir_path = path + 'ir/'
depth_path = path + 'depth/'

rgb_images = [rgb_path + p for p in os.listdir(rgb_path)]
ir_images = [ir_path + p for p in os.listdir(ir_path)]
depth_images = [depth_path + p for p in os.listdir(depth_path)]

ssim_pd = pd.read_csv('ssim.csv')

for idx, (rgb,ir,depth) in enumerate(zip(rgb_images,ir_images,depth_images)):
    rgb_im = Image.open(rgb).convert('RGB')
    ir_im = Image.open(ir).convert('RGB')
    depth_im = Image.open(depth).convert('RGB')
    RGB, IR, DEPTH = transform(rgb_im), transform(ir_im), transform(depth_im)
    score = compute_edge_border_iou(RGB,IR)
    # print(f'{idx:04d} image ssim = {ssim(RGB,IR):.4f}')
    print(f'{idx:04d} image IOU = {compute_edge_border_iou(RGB,IR):.4f}')

    ssim_pd.loc[idx,'iou'] = f'{compute_edge_border_iou(RGB,IR):.4f}'
ssim_pd.to_csv('ssim.csv')