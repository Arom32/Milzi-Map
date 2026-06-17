import cv2
import numpy as np


class HeatmapGenerator:
    """Render a density map as an overlay on the source image."""

    def __init__(self, alpha=0.6, colormap=cv2.COLORMAP_JET):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.colormap = colormap

    def render(self, image, density_map, virtual_to_src_matrix):
        height, width = image.shape[:2]
        warped_density = cv2.warpPerspective(
            density_map,
            virtual_to_src_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        density_norm = self._normalize_density(warped_density, (height, width))
        heatmap = cv2.applyColorMap(density_norm, self.colormap)
        alpha_mask = (density_norm.astype(np.float32) / 255.0)[:, :, None] * self.alpha
        result = (1.0 - alpha_mask) * image + alpha_mask * heatmap
        return np.clip(result, 0, 255).astype(np.uint8)

    def _normalize_density(self, density, shape):
        if density.max() <= 0:
            return np.zeros(shape, dtype=np.uint8)

        return cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
