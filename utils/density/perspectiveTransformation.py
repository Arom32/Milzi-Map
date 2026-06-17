
import cv2
import numpy as np

class PerspectiveTransformer:
    """이미지 공간과 가상 바닥 평면 공간 사이의 좌표 변환을 담당"""
    MIN_VIRTUAL_W = 400
    MAX_VIRTUAL_W = 900

    def __init__(self, image_shape: tuple, axis_config: dict = None):
        self.image_shape = image_shape
        self.axis_config = axis_config
        self.virtual_w = 0
        self.virtual_h = 0
        self.src_to_virtual_matrix = None
        self.virtual_to_src_matrix = None
        
        self._initialize_matrices()

    def _initialize_matrices(self):
        if not self.axis_config:
            height, width = self.image_shape[:2]
            self.virtual_w, self.virtual_h = width, height
            self.src_to_virtual_matrix = np.eye(3, dtype=np.float32)
            self.virtual_to_src_matrix = np.eye(3, dtype=np.float32)
            return

        if self.axis_config.get("mode") != "floor_lines":
            raise ValueError("Unsupported perspective correction mode.")

        src_points = np.asarray(self.axis_config["src_points"], dtype=np.float32)
        if src_points.shape != (4, 2):
            raise ValueError("Floor line input must provide four source points.")

        # 가상 평면의 해상도 결정
        top_w = np.linalg.norm(src_points[1] - src_points[0])
        bottom_w = np.linalg.norm(src_points[2] - src_points[3])
        left_h = np.linalg.norm(src_points[3] - src_points[0])
        right_h = np.linalg.norm(src_points[2] - src_points[1])

        raw_w = max(top_w, bottom_w, 1.0)
        raw_h = max(left_h, right_h, 1.0)
        
        self.virtual_w = int(np.clip(raw_w, self.MIN_VIRTUAL_W, self.MAX_VIRTUAL_W))
        self.virtual_h = max(1, int(round(self.virtual_w * (raw_h / raw_w))))

        dst_points = np.array([
            [0, 0],
            [self.virtual_w - 1, 0],
            [self.virtual_w - 1, self.virtual_h - 1],
            [0, self.virtual_h - 1]
        ], dtype=np.float32)

        self.src_to_virtual_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.virtual_to_src_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

    def project_boxes_to_plane(self, bboxes: np.ndarray) -> np.ndarray:
        """박스의 하단 중앙점을 가상 평면 좌표로 투영"""
        if len(bboxes) == 0:
            return np.empty((0, 2), dtype=np.float32)

        x1, _, x2, y2 = bboxes.T
        bottom_centers = np.column_stack(((x1 + x2) * 0.5, y2)).astype(np.float32)
        
        return cv2.perspectiveTransform(
            bottom_centers.reshape(-1, 1, 2), self.src_to_virtual_matrix
        ).reshape(-1, 2)

    def get_valid_mask(self, points: np.ndarray) -> np.ndarray:
        """가상 평면 영역 내부에 존재하는 점들의 마스크 반환"""
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        return (
            (points[:, 0] >= 0) & (points[:, 0] < self.virtual_w) &
            (points[:, 1] >= 0) & (points[:, 1] < self.virtual_h)
        )
