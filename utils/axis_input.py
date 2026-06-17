import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


LINE_X_COLOR = "#ff4b4b"
LINE_Y_COLOR = "#4b9eff"
CANVAS_HEIGHT = 500


def render_axis_input(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = image.size
    canvas_w = max(1, int(orig_w * CANVAS_HEIGHT / orig_h))
    scale_x = orig_w / canvas_w
    scale_y = orig_h / CANVAS_HEIGHT

    with st.expander("Floor perspective lines", expanded=False):
        use_lines = st.checkbox("Use floor perspective correction", value=False)
        if not use_lines:
            st.caption("If no lines are set, the full image coordinate system is used.")
            return None

        st.caption(
            "Draw two red lines in the floor X direction and two blue lines in the floor Y direction."
        )
        draw_mode = st.radio(
            "Line to draw",
            ["X direction", "Y direction"],
            horizontal=True,
            label_visibility="collapsed",
        )
        stroke_color = LINE_X_COLOR if draw_mode == "X direction" else LINE_Y_COLOR

        canvas_result = st_canvas(
            background_image=image,
            drawing_mode="line",
            stroke_color=stroke_color,
            stroke_width=3,
            width=canvas_w,
            height=CANVAS_HEIGHT,
            key="floor_lines_canvas",
        )

        lines = _extract_scaled_lines(canvas_result, scale_x, scale_y)
        x_lines, y_lines = _select_direction_lines(lines)

        col_x, col_y = st.columns(2)
        with col_x:
            st.success(f"X lines: {len(x_lines)}/2") if len(x_lines) >= 2 else st.info(
                f"Draw red X-direction lines: {len(x_lines)}/2"
            )
        with col_y:
            st.success(f"Y lines: {len(y_lines)}/2") if len(y_lines) >= 2 else st.info(
                f"Draw blue Y-direction lines: {len(y_lines)}/2"
            )

        if len(x_lines) < 2 or len(y_lines) < 2:
            return None

        floor_quad = _floor_quad_from_lines(x_lines[:2], y_lines[:2])
        if floor_quad is None:
            st.warning("The selected lines do not form a valid floor quadrilateral.")
            return None

        st.caption("Floor quadrilateral ready.")
        return {
            "mode": "floor_lines",
            "x_lines": x_lines[:2],
            "y_lines": y_lines[:2],
            "src_points": floor_quad,
        }


def _extract_scaled_lines(canvas_result, scale_x, scale_y):
    if canvas_result is None or canvas_result.json_data is None:
        return []

    lines = []
    for obj in canvas_result.json_data.get("objects", []):
        if obj.get("type") != "line":
            continue

        x1 = float(obj["left"])
        y1 = float(obj["top"])
        x2 = float(obj["left"] + obj["width"])
        y2 = float(obj["top"] + obj["height"])
        lines.append(
            {
                "p1": (x1 * scale_x, y1 * scale_y),
                "p2": (x2 * scale_x, y2 * scale_y),
                "color": obj.get("stroke", "").lower(),
            }
        )

    return lines


def _select_direction_lines(lines):
    x_lines = []
    y_lines = []
    for line in lines:
        if line["color"] == LINE_X_COLOR:
            x_lines.append(line)
        elif line["color"] == LINE_Y_COLOR:
            y_lines.append(line)
    return x_lines, y_lines


def _floor_quad_from_lines(x_lines, y_lines):
    intersections = []
    for x_line in x_lines:
        row = []
        for y_line in y_lines:
            point = _line_intersection(x_line, y_line)
            if point is None:
                return None
            row.append(point)
        intersections.append(row)

    points = np.array(
        [intersections[0][0], intersections[0][1], intersections[1][1], intersections[1][0]],
        dtype=np.float32,
    )
    return _order_quad_clockwise(points).tolist()


def _order_quad_clockwise(points):
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]

    start_index = np.argmin(ordered.sum(axis=1))
    ordered = np.roll(ordered, -start_index, axis=0)

    signed_area = _polygon_signed_area(ordered)
    if signed_area < 0:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)

    return ordered


def _polygon_signed_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _line_intersection(line_a, line_b):
    a = np.array(line_a["p1"], dtype=np.float32)
    b = np.array(line_a["p2"], dtype=np.float32)
    c = np.array(line_b["p1"], dtype=np.float32)
    d = np.array(line_b["p2"], dtype=np.float32)

    d1 = b - a
    d2 = d - c
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(float(cross)) < 1e-6:
        return None

    t = ((c[0] - a[0]) * d2[1] - (c[1] - a[1]) * d2[0]) / cross
    return a + t * d1
