import numpy as np
import cv2
import utils.config as config

class DensityEstimator : 
    def __init__(self, sigma = config.DEFAULT_GAUSSIAN_SIGMA):
        self.min_sigma = config.SIGMA_MIN
        self.max_sigma = config.SIGMA_MAX
        self.base_sigma_ratio = sigma/self.max_sigma


    def estimate(self, image_shape: tuple, bboxes: np.ndarray) -> np.ndarray:
        """
        바운딩 박스를 입력받아 2D 밀집도 행렬 반환
        """
        height, width = image_shape[:2]
        density_map = np.zeros((height, width), dtype=np.float32)

        if len(bboxes) == 0:
            return density_map

        for box in bboxes:
            x1, y1, x2, y2 = box[:4]
            # Center 추출
            cx, cy = int(x1 + x2) >> 1, int(y1 + y2) >> 1  
            
            # bb크기 반영 시그마 계산 
            sigma = (((x2 - x1) + (y2 - y1)) / 2) * self.base_sigma_ratio
            sigma = np.clip(sigma, self.min_sigma, self.max_sigma)
            
            # 가우시안 커널 생성 
            kernel_size = int(sigma * 6) | 1 
            kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
            kernel_2d = kernel_1d @ kernel_1d.T
            # 1d == N by 1 matrix 
            # 2d (N,1)*(1,N) : N by N matrix

            # 정규화
            kernel_2d = kernel_2d / kernel_2d.max()
            
            # 경계선 클리핑
            half = kernel_size >> 1 
            y_start, y_end = max(0, cy - half), min(height, cy + half + 1)
            x_start, x_end = max(0, cx - half), min(width, cx + half + 1)
            
            if y_start >= y_end or x_start >= x_end:
                continue
                
            k_y_start = half - (cy - y_start)
            k_y_end   = half + (y_end - cy)
            k_x_start = half - (cx - x_start)
            k_x_end   = half + (x_end - cx)
            
            
            # 밀집도 누적 
            density_map[y_start:y_end, x_start:x_end] += kernel_2d[k_y_start:k_y_end, k_x_start:k_x_end]
            

        return density_map