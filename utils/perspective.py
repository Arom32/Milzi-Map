
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from sklearn.cluster import DBSCAN
from utils.affine import get_inverse_transform_matrix, transform_bboxes_to_local

LINE_X_COLOR = "#FF4B4B"
LINE_Y_COLOR = "#4B9EFF"
CANVAS_HEIGHT = 500

def extract_lines_from_canvas(canvas_data) -> list[dict]:
    lines = []
    if canvas_data is None or canvas_data.json_data is None or "objects" not in canvas_data.json_data:
        return lines
    for obj in canvas_data.json_data["objects"]:
        if obj.get("type") != "line": continue
        x1, y1 = obj["left"], obj["top"]
        x2, y2 = obj["left"] + obj["width"], obj["top"] + obj["height"]
        color = obj.get("stroke", "#ffffff")
        lines.append({"p1": (x1, y1), "p2": (x2, y2), "color": color})
    return lines

def scale_line(line: dict, scale_x: float, scale_y: float) -> dict:
    x1, y1 = line["p1"]
    x2, y2 = line["p2"]
    return {
        "p1": (x1 * scale_x, y1 * scale_y),
        "p2": (x2 * scale_x, y2 * scale_y),
        "color": line["color"],
    }

def parse_axis_lines(lines: list[dict]) -> tuple:
    line_x, line_y = None, None
    for line in lines:
        if LINE_X_COLOR.lower() in line["color"].lower(): line_x = line
        elif LINE_Y_COLOR.lower() in line["color"].lower(): line_y = line
    return line_x, line_y

def line_intersection(line_a: dict, line_b: dict) -> np.ndarray | None:
    A, B = np.array(line_a["p1"], float), np.array(line_a["p2"], float)
    C, D = np.array(line_b["p1"], float), np.array(line_b["p2"], float)
    d1, d2 = B - A, D - C
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-8: return None
    t = ((C[0] - A[0]) * d2[1] - (C[1] - A[1]) * d2[0]) / cross
    return A + t * d1

def render_line_input(uploaded_file) -> dict | None:
    img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = img.size
    canvas_w = int(orig_w * CANVAS_HEIGHT / orig_h)
    scale_x, scale_y = orig_w / canvas_w, orig_h / CANVAS_HEIGHT

    st.markdown("#### 원근 보정 축 설정")
    col1, col2 = st.columns([1, 1])
    with col1: st.caption("아래에서 그릴 축을 선택하고 이미지 위에 직선을 그리세요.")
    with col2:
        st.markdown(
            f"<span style='color:{LINE_X_COLOR}'>■</span> **X축** (좌우) "
            f"<span style='color:{LINE_Y_COLOR}'>■</span> **Y축** (깊이)",
            unsafe_allow_html=True,
        )

    draw_mode = st.radio("그릴 축 선택", ["X축 (좌우)", "Y축 (깊이)"], horizontal=True, label_visibility="collapsed")
    current_color = LINE_X_COLOR if draw_mode == "X축 (좌우)" else LINE_Y_COLOR

    canvas_result = st_canvas(
        background_image=img, drawing_mode="line", stroke_color=current_color,
        stroke_width=3, width=canvas_w, height=CANVAS_HEIGHT, key="main_canvas",
    )

    lines = extract_lines_from_canvas(canvas_result)
    raw_line_x, raw_line_y = parse_axis_lines(lines)

    result_x = scale_line(raw_line_x, scale_x, scale_y) if raw_line_x else None
    result_y = scale_line(raw_line_y, scale_x, scale_y) if raw_line_y else None

    col_status1, col_status2 = st.columns(2)
    with col_status1: st.success("X축 입력 완료") if result_x else st.info("X축을 그려주세요.")
    with col_status2: st.success("Y축 입력 완료") if result_y else st.info("Y축을 그려주세요.")

    if result_x is not None and result_y is not None:
        origin = line_intersection(result_x, result_y)
        if origin is None:
            st.warning("두 직선이 평행합니다. 다시 그려주세요.")
            return None
        st.markdown("---")
        st.markdown(f"**기준점 계산 완료:** `({int(origin[0])}, {int(origin[1])})`")
        return {"line_x": result_x, "line_y": result_y, "origin": (float(origin[0]), float(origin[1]))}
    return None


def apply_perspective_and_cluster(bboxes: list | np.ndarray, axis_config: dict, eps: float = 2.0, min_samples: int = 3) -> tuple:
    if len(bboxes) == 0 or not axis_config:
        return np.array([]), np.array([])

    # affine.py에 위임하여 좌표계 변환 수행
    M_inv, origin = get_inverse_transform_matrix(axis_config)
    transformed_points = transform_bboxes_to_local(bboxes, M_inv, origin)

    # DBSCAN 기반 군집화 수행
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(transformed_points)
    
    return transformed_points, clustering.labels_