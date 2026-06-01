''' 
1. roi 설정, 카메라 각도로 인한 왜곡 보정
    - 불필요한 연산 제거
    - 검출 정확도 확보
    - 좌표계 변환
2. roi 기준 타일 분할
    - 그리드 분할
    - yolo는 bb의 중심점 좌표를 예측 하므로, 정수리기준으로 객체 탐지가능 할 것으로 예상

3. heatmap 생성
    - 타일별 객체 탐지 결과로 히트맵 생성
    - 타일별로 객체수 카운트
    - 색상으로 밀집도 시각화
4. 히트맵 오버레이
    - 좌표계 변환 - 원본이미지로 
    - 원본이미지에 히트맵 오버레이
    - 알파블렌딩?

'''
from email.mime import image

import cv2
import numpy as np

class HeatmapGenerator:

    def __init__(self, image, bboxes, src_points, grid_shape=(10, 10), sigma=80, alpha=0.5):
        self.raw_image = image
        self.src_points = src_points
        self.alpha = alpha
        self.bboxes = bboxes
        self.sigma = sigma

        self.grid_rows = max(1, grid_shape[0])
        self.grid_cols = max(1, grid_shape[1])

        self.virtual_w = 0
        self.virtual_h = 0
        self.virtual_to_src_matrix = None
        self.src_to_virtual_matrix = None

        self.overlay_img = None
        # 실시간 FPS 방어를 위한 최적 스케일 한계점
        self.MIN_VIRTUAL_W = 400
        self.MAX_VIRTUAL_W = 800  
         
        self.scale_x , self.scale_y = image.shape[1], image.shape[0]

    def undistortion(self):
        # 사다리꼴 ROI의 실제 비율을 계산하여 가상의 직사각형 공간으로 매핑,
        # src_points: 원본 이미지 내 사다리꼴 ROI 네 꼭지점 (4, 2) - [좌상, 우상, 우하, 좌하] 순서
        # grid_shape: (grid_rows, grid_cols)
        
        src_pts = np.float32(self.src_points)
        # 왜곡 보정
        # roi 공간(사다리꼴) -> 직사각형으로 매핑 // roi 공간을 늘리고 줄여서 직사각형 꼴로 만듦
        # 사다리꼴의 실제 종횡비 이용해 사각형 해상도 설정

        # 가로
        width_top = np.linalg.norm(src_pts[0] - src_pts[1])
        width_bottom = np.linalg.norm(src_pts[3] - src_pts[2])
        raw_w = int(max(width_top, width_bottom))
        
        # 세로
        height_left = np.linalg.norm(src_pts[0] - src_pts[3])
        height_right = np.linalg.norm(src_pts[1] - src_pts[2])
        raw_h = int(max(height_left, height_right))
        
        #제로 디비전 방지 예외 처리
        if raw_w == 0 or raw_h == 0:
            raw_w, raw_h = 800, 800

        #Aspect Ratio 유지
        aspect_ratio = raw_h / raw_w

        # 연산 최적화를 위한 크기 임계치 설정
        # 가로 크기를 제한 범위 내로 클램핑
        virtual_w = int(max(self.MIN_VIRTUAL_W, min(raw_w, self.MAX_VIRTUAL_W)))
        virtual_h = int(virtual_w * aspect_ratio)

        # 2. 가상 직사각형(virtual_w x virtual_h) 좌표 정의 및 변환 행렬 산출
        dst_points = np.array([
            [0, 0], 
            [virtual_w, 0], 
            [virtual_w, virtual_h], 
            [0, virtual_h]
        ], dtype=np.float32)
        
        # https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga20f62aa3235d869c9956436c870893ae
        # perspective transform 행렬 계산, 실제로 사라디꼴 roi영역을 직사각형으로 변형
        self.src_to_virtual_matrix = cv2.getPerspectiveTransform(src_pts, dst_points)

        # linalg, 역행렬 연산 raw이미지에서 vir로 변환한 행렬의 역연산, 다시 원본 이미지로 변환시 사용
        self.virtual_to_src_matrix = np.linalg.inv(self.src_to_virtual_matrix) 
        
    def grid_counting_and_heatmap_generation(self):
        cell_w = self.virtual_w / self.grid_cols
        cell_h = self.virtual_h / self.grid_rows
        grid_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        
        if len(self.bboxes) > 0:
            # 객체 중심점 할당 및 Homogeneous Coordinates변환, 실제 객체 위치를 가상 직사각형 내의 좌표로 변환
            pts = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in self.bboxes], dtype=np.float32)
            ones = np.ones((pts.shape[0], 1), dtype=np.float32)
            pts_homo = np.hstack([pts, ones])
            
            # 행렬 곱 연산 및 정규화
            trans_pts = np.dot(self.src_to_virtual_matrix, pts_homo.T).T
            trans_pts /= trans_pts[:, 2:3]
            
            for px, py in trans_pts[:, :2]:
                if 0 <= px < self.virtual_w and 0 <= py < self.virtual_h:
                    c = min(int(px // cell_w), self.grid_cols - 1)
                    r = min(int(py // cell_h), self.grid_rows - 1)
                    grid_counts[r, c] += 1.0

    def draw_density_heatmap(self):
        # 사진 내부의 공간 고려 없이 단순 bounding box에 대한 
        h, w = self.raw_image.shape[:2]
        # 누적(Accumulation) 연산을 위해 float32 타입으로 0 행렬 초기화
        density_map = np.zeros((h, w), dtype=np.float32) 

        # 2D 가우시안 커널(Gaussian Kernel) 생성
        k_size = int(self.sigma * 6) | 1  # 항상 홀수 유지
        offset = k_size // 2
        
        kernel_1d = cv2.getGaussianKernel(k_size, self.sigma)
        kernel_2d = kernel_1d @ kernel_1d.T
        # 중심 피크(Peak)를 1.0으로 정규화하여 겹침 횟수에 비례해 직관적으로 누적되도록 처리
        kernel_2d = kernel_2d / np.max(kernel_2d) 

        # 2. 좌표 변환 및 선형 누적(Linear Accumulation)
        for box in self.bboxes:
            x1, y1, x2, y2 = box
            
            # 해상도 불일치 방지를 위한 스케일(Scale) 반영
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 이미지 경계(Boundary)를 벗어나지 않도록 클리핑(Clipping)
            y_min, y_max = max(0, cy - offset), min(h, cy + offset + 1)
            x_min, x_max = max(0, cx - offset), min(w, cx + offset + 1)

            # 이미지 경계에 걸치는 경우 커널에서 사용할 부분만 추출
            k_y_min, k_y_max = max(0, offset - cy), k_size - max(0, (cy + offset + 1) - h)
            k_x_min, k_x_max = max(0, offset - cx), k_size - max(0, (cx + offset + 1) - w)

            if y_min >= y_max or x_min >= x_max:
                continue

            # 겹치는 영역에 대해 커널 값을 선형 누적
            density_map[y_min:y_max, x_min:x_max] += kernel_2d[k_y_min:k_y_max, k_x_min:k_x_max]

        # 3. 최소-최대 정규화(Min-Max Normalization)
        if np.max(density_map) > 0:
            density_map_norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        else:
            density_map_norm = density_map
            
        density_map_norm = density_map_norm.astype(np.uint8)

        # 4. 컬러맵(Colormap) 적용 및 동적 알파 블렌딩(Dynamic Alpha Blending)
        heatmap = cv2.applyColorMap(density_map_norm, cv2.COLORMAP_JET)
        
        # 밀도 맵을 0.0 ~ 1.0 사이의 가중치 마스크(Weight Mask)로 활용
        # 밀도가 높은 곳(붉은색)일수록 히트맵을 강하게 표시하고, 밀도가 0인 곳은 원본 유지
        alpha_mask = density_map_norm.astype(np.float32) / 255.0
        alpha_mask = np.expand_dims(alpha_mask, axis=2) 
        
        # 합성 가중치 (최대 히트맵 투명도 한계: 0.6)
        heatmap_weight = alpha_mask * 0.6
        image_weight = 1.0 - heatmap_weight
        
        result_image = (image_weight * self.raw_image) + (heatmap_weight * heatmap)
        result_image = np.clip(result_image, 0, 255).astype(np.uint8)

        return result_image

