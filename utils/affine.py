"""
https://kr.mathworks.com/discovery/affine-transformation.html
Affine Transformation 및 Coordinate Projection 모듈
"""

import numpy as np

def get_inverse_transform_matrix(axis_config: dict) -> tuple:
    """
    X축, Y축 직선을 기반으로 역변환 행렬과 원점(Origin)을 반환.
    """
    origin = np.array(axis_config["origin"])
    
    # 단위 벡터(Unit Vector) 계산
    vec_x = np.array(axis_config["line_x"]["p2"]) - np.array(axis_config["line_x"]["p1"])
    vec_x = vec_x / np.linalg.norm(vec_x)

    vec_y = np.array(axis_config["line_y"]["p2"]) - np.array(axis_config["line_y"]["p1"])
    vec_y = vec_y / np.linalg.norm(vec_y)

    # 변환 행렬 구성
    M = np.column_stack((vec_x, vec_y))
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        raise ValueError("X축과 Y축이 선형 종속(Linearly Dependent) 상태여서 역행렬을 계산할 수 없습니다.")
        
    return M_inv, origin

def transform_bboxes_to_local(bboxes: list | np.ndarray, M_inv: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """
    바운딩 박스의 하단 중앙(Bottom-Center) 픽셀 좌표를 로컬 좌표계로 투영.
    수식: $P_{local} = M^{-1}(P_{pixel} - Origin)$
    """
    transformed_points = []
    for box in bboxes:
        x1, y1, x2, y2 = box[:4]
        bottom_center = np.array([(x1 + x2) / 2, y2])
        local_point = M_inv.dot(bottom_center - origin)
        transformed_points.append(local_point)

    return np.array(transformed_points)