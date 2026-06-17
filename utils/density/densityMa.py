from dataclasses import dataclass

import cv2
import numpy as np
from utils.density.densityMa import PerspectiveTransformer 
from Clusterer import PointClusterer

class DensityResult:
    density_map: np.ndarray
    virtual_to_src_matrix: np.ndarray
    grid_counts: np.ndarray
    grid_density: np.ndarray
    points: np.ndarray
    valid_points: np.ndarray
    labels: np.ndarray
    cluster_count: int
    noise_count: int


class DensityMapGenerator:
    """그리드 카운팅 및 가우시안 KDE 맵 생성을 담당"""
    def __init__(self, grid_shape: tuple = (10, 10), sigma: float = 80.0):
        self.grid_rows = max(1, int(grid_shape[0]))
        self.grid_cols = max(1, int(grid_shape[1]))
        self.sigma = max(1.0, float(sigma))

    def generate_grid_stats(self, points: np.ndarray, w: int, h: int) -> tuple:
        """그리드 별 빈도수 및 면적 대비 밀도 계산"""
        grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        cell_w, cell_h = w / self.grid_cols, h / self.grid_rows

        if len(points) > 0:
            for px, py in points:
                col = min(int(px // cell_w), self.grid_cols - 1)
                row = min(int(py // cell_h), self.grid_rows - 1)
                grid_counts[row, col] += 1.0

        grid_density = grid_counts / max(cell_w * cell_h, 1.0)
        return grid_counts, grid_density

    def generate_kde_map(self, points: np.ndarray, labels: np.ndarray, w: int, h: int) -> np.ndarray:
        """클러스터 크기 가중치가 반영된 가우시안 KDE 밀도 맵 생성"""
        density = np.zeros((h, w), dtype=np.float32)
        if len(points) == 0:
            return density

        # 가우시안 커널 빌드
        sigma_int = max(1, int(round(self.sigma)))
        kernel_size = max(3, int(sigma_int * 6) | 1)
        offset = kernel_size // 2
        kernel_1d = cv2.getGaussianKernel(kernel_size, sigma_int)
        kernel = kernel_1d @ kernel_1d.T
        kernel = kernel / max(float(kernel.max()), 1e-6)

        # 클러스터 크기 별 가중치 매핑
        cluster_sizes = {}
        for label in labels:
            if label >= 0:
                cluster_sizes[int(label)] = cluster_sizes.get(int(label), 0) + 1

        # 맵에 가우시안 스플래팅(Splatting) 수행
        for point, label in zip(points, labels):
            cx, cy = int(round(point[0])), int(round(point[1]))
            weight = 1.0 + np.log1p(cluster_sizes[int(label)]) if label >= 0 else 1.0

            y_min, y_max = max(0, cy - offset), min(h, cy + offset + 1)
            x_min, x_max = max(0, cx - offset), min(w, cx + offset + 1)
            if y_min >= y_max or x_min >= x_max:
                continue

            k_y_min = max(0, offset - cy)
            k_y_max = kernel_size - max(0, (cy + offset + 1) - h)
            k_x_min = max(0, offset - cx)
            k_x_max = kernel_size - max(0, (cx + offset + 1) - w)

            density[y_min:y_max, x_min:x_max] += kernel[k_y_min:k_y_max, k_x_min:k_x_max] * weight

        return density



class DensityEstimator:
    """각 컴포넌트의 책임을 조율하여 최종 DensityResult를 빌드하는 Facade 엔티티"""
    def __init__(self, image_shape, axis_config=None, grid_shape=(10, 10), sigma=80.0, dbscan_eps=None, dbscan_min_samples=3):
        self.transformer = PerspectiveTransformer(image_shape, axis_config)
        self.clusterer = PointClusterer(dbscan_eps, dbscan_min_samples)
        self.map_generator = DensityMapGenerator(grid_shape, sigma)

    def estimate(self, bboxes) -> DensityResult:
        boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        
        # 1. 좌표 투영 및 유효성 필터링
        all_points = self.transformer.project_boxes_to_plane(boxes)
        valid_mask = self.transformer.get_valid_mask(all_points)
        valid_points = all_points[valid_mask]

        # 2. 통계 및 밀도 계산
        v_w, v_h = self.transformer.virtual_w, self.transformer.virtual_h
        grid_counts, grid_density = self.map_generator.generate_grid_stats(valid_points, v_w, v_h)

        # 3. 클러스터링 라벨 계산
        # 내부 DBSCAN 자동 eps 세팅 로직을 조율층에서 계산하여 전달
        default_eps = max(v_w / self.map_generator.grid_cols, v_h / self.map_generator.grid_rows) * 1.2
        labels = self.clusterer.compute_labels(valid_points, default_eps)

        # 4. 가우시안 KDE 맵 빌드
        density_map = self.map_generator.generate_kde_map(valid_points, labels, v_w, v_h)

        # 5. 메타데이터 요약 및 결과 반환
        cluster_labels = {int(label) for label in labels if label >= 0}
        noise_count = int(np.sum(labels == -1)) if len(labels) else 0

        return DensityResult(
            density_map=density_map,
            virtual_to_src_matrix=self.transformer.virtual_to_src_matrix,
            grid_counts=grid_counts,
            grid_density=grid_density,
            points=all_points,
            valid_points_mask=valid_mask,
            labels=labels,
            cluster_count=len(cluster_labels),
            noise_count=noise_count,
        )