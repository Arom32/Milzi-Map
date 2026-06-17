import streamlit as st
from PIL import Image

import utils.config as config
from utils.axis_input import render_axis_input
from utils.processor import VisionPipeline


st.set_page_config(page_title=config.PAGE_TITLE, layout=config.PAGE_LAYOUT)
st.title(config.MAIN_TITLE)

st.sidebar.header("YOLOv8 model")
selected_model = st.sidebar.selectbox("Detection model", config.AVAILABLE_MODELS)

if (
    "pipeline" not in st.session_state
    or st.session_state.get("current_model") != selected_model
):
    with st.spinner(f"Loading model '{selected_model}'..."):
        st.session_state.pipeline = VisionPipeline(model_path=selected_model)
        st.session_state.current_model = selected_model

pipeline = st.session_state.pipeline

st.sidebar.header("Analysis settings")
conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    config.CONF_MIN,
    config.CONF_MAX,
    config.DEFAULT_CONF_THRESHOLD,
    config.CONF_STEP,
)
grid_rows = st.sidebar.slider(
    "Grid rows",
    config.GRID_MIN,
    config.GRID_MAX,
    config.DEFAULT_GRID_ROWS,
)
grid_cols = st.sidebar.slider(
    "Grid columns",
    config.GRID_MIN,
    config.GRID_MAX,
    config.DEFAULT_GRID_COLS,
)
gaussian_sigma = st.sidebar.slider(
    "Gaussian sigma",
    config.SIGMA_MIN,
    config.SIGMA_MAX,
    config.DEFAULT_GAUSSIAN_SIGMA,
    config.SIGMA_STEP,
)
grid_shape = (grid_rows, grid_cols)

main_menu = st.radio(
    "Analysis mode",
    ["Density", "Flow"],
    horizontal=True,
    label_visibility="collapsed",
)

if main_menu == "Density":
    uploaded_file = st.file_uploader(
        "Upload image (jpg, png)",
        type=["jpg", "jpeg", "png"],
        key="file_uploader",
    )
    
    # 보정 축
    axis_config = render_axis_input(uploaded_file) if uploaded_file else None
    st.markdown("---")

    view_col, data_col = st.columns([0.65, 0.35])

    with view_col:
        tab_bb, tab_heatmap, tab_raw = st.tabs(
            ["Bounding Box", "Density Heatmap", "Original"]
        )

        if uploaded_file:
            with st.spinner("Analyzing image..."):
                try:
                    res_bb, res_heatmap, count, density_result = pipeline.process_all_views(
                        input_file=uploaded_file,
                        conf_threshold=conf_threshold,
                        grid_shape=grid_shape,
                        gaussian_sigma=gaussian_sigma,
                        axis_config=axis_config,
                    )

                    with data_col:
                        valid_count = int(density_result.valid_points.sum())
                        max_cell_count = int(density_result.grid_counts.max())

                        st.subheader("Results")
                        st.metric(label="Detected people", value=f"{count:03d}")
                        st.metric(label="Valid local points", value=f"{valid_count:03d}")
                        st.metric(label="Max people in cell", value=f"{max_cell_count:03d}")
                        st.metric(label="Clusters", value=f"{density_result.cluster_count:03d}")
                        st.metric(label="Noise points", value=f"{density_result.noise_count:03d}")

                        if count >= config.DANGER_THRESHOLD:
                            st.error("Risk: high")
                        elif count >= config.WARNING_THRESHOLD:
                            st.warning("Risk: medium")
                        else:
                            st.success("Risk: low")

                    with tab_bb:
                        st.image(
                            res_bb,
                            caption="YOLOv8 bounding box result",
                            use_column_width=True,
                        )

                    with tab_heatmap:
                        caption = (
                            "Floor-perspective density heatmap"
                            if axis_config
                            else "Full-image density heatmap"
                        )
                        st.image(res_heatmap, caption=caption, use_column_width=True)

                    with tab_raw:
                        uploaded_file.seek(0)
                        st.image(
                            Image.open(uploaded_file),
                            caption="Uploaded original image",
                            use_column_width=True,
                        )

                except Exception as exc:
                    st.error(f"Failed to process image: {exc}")
                    st.exception(exc)
        else:
            with tab_bb:
                st.info("Upload an image to show bounding boxes.")
            with tab_heatmap:
                st.info("Upload an image to show the density heatmap.")
            with tab_raw:
                st.info("Upload an image to show the original.")

elif main_menu == "Flow":
    st.info("Flow estimation and tracking is planned.")
