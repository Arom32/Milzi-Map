import cv2
import numpy as np


class HeatmapGenerator:
    """밀집도 행렬을 컬러맵으로 변환"""

    def __init__(self, alpha=0.6, colormap=cv2.COLORMAP_TURBO):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.colormap = colormap

    def render(self, image: np.ndarray, density_map: np.ndarray, global_max=None) -> np.ndarray:
        """
        density_map: density.py에서 생성된 2D 행렬
        global_max: 일관된 색상 스케일을 위한 절대 최대 밀집도 기준값
        """
        if density_map.max() <= 0:
            return image.copy()

        # Normalization
        if global_max is not None and global_max > 0:
            density_norm = np.clip((density_map / global_max) * 255.0, 0, 255).astype(np.uint8)
        else:
            density_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Color Mapping
        heatmap = cv2.applyColorMap(density_norm, self.colormap)
        
        #  Alpha Blending
        alpha_mask = (density_norm.astype(np.float32) / 255.0)[:, :, None] * self.alpha
        result = (1.0 - alpha_mask) * image + alpha_mask * heatmap
        
        return np.clip(result, 0, 255).astype(np.uint8)
