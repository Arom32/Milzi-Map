import numpy as np

class PointClusterer:
    """평면상의 점들을 공간적으로 군집화 (DBSCAN 구현)"""
    def __init__(self, eps: float = None, min_samples: int = 3):
        self.eps = eps
        self.min_samples = max(1, int(min_samples))

    def compute_labels(self, points: np.ndarray, default_eps: float) -> np.ndarray:
        if len(points) == 0:
            return np.empty(0, dtype=int)
        if len(points) < self.min_samples:
            return np.full(len(points), -1, dtype=int)

        eps = self.eps if self.eps is not None else default_eps

        # 벡터화된 거리 행렬 계산 및 이웃 탐색
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        neighbors = [np.flatnonzero(row <= eps) for row in distances]
        
        labels = np.full(len(points), -1, dtype=int)
        visited = np.zeros(len(points), dtype=bool)
        cluster_id = 0

        for idx in range(len(points)):
            if visited[idx]:
                continue

            visited[idx] = True
            if len(neighbors[idx]) < self.min_samples:
                continue

            labels[idx] = cluster_id
            seeds = list(neighbors[idx])
            seed_cursor = 0

            while seed_cursor < len(seeds):
                seed = seeds[seed_cursor]
                if not visited[seed]:
                    visited[seed] = True
                    if len(neighbors[seed]) >= self.min_samples:
                        for candidate in neighbors[seed]:
                            if candidate not in seeds:
                                seeds.append(int(candidate))

                if labels[seed] == -1:
                    labels[seed] = cluster_id
                seed_cursor += 1

            cluster_id += 1

        return labels

