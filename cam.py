# cam.py - Grad-CAM reshape + heatmap utils for Swin Transformer
import numpy as np
import cv2


def reshape_transform(tensor, height=None, width=None):
    """
    Reshape Swin Transformer output tensor to match CNN-like format for CAM.
    Args:
        tensor: torch.Tensor of shape [B, N, C]
        height: int, height of feature map (e.g., 24)
        width: int, width of feature map (e.g., 24)
    Returns:
        reshaped tensor: [B, C, H, W]
    """
    if height is None or width is None:
        # Auto-infer square shape
        spatial_dim = int((tensor.size(1)) ** 0.5)
        height = width = spatial_dim

    reshaped = tensor.view(tensor.size(0), height, width, tensor.size(2))  # [B, H, W, C]
    return reshaped.permute(0, 3, 1, 2)  # [B, C, H, W]


def heatmap_filter(cam, img):
    """
    Generate heatmap and overlay CAM result.
    Args:
        cam: ndarray [H, W], values [0~1]
        img: original image (OpenCV BGR)
    Returns:
        raw CAM, heatmap, overlay CAM
    """
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))  # Resize CAM to original image size
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img) / 255
    cam_img = cam_img / np.max(cam_img)
    return cam_resized, np.uint8(255 * heatmap), np.uint8(255 * cam_img)
