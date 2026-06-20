import streamlit as st
from PIL import Image

import utils.config as config
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

# 사이드 바 

st.sidebar.header("Analysis settings")
conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    config.CONF_MIN,
    config.CONF_MAX,
    config.DEFAULT_CONF_THRESHOLD,
    config.CONF_STEP,
)
gaussian_sigma = st.sidebar.slider(
    "Gaussian sigma",
    config.SIGMA_MIN,
    config.SIGMA_MAX,
    config.DEFAULT_GAUSSIAN_SIGMA,
    config.SIGMA_STEP,
)
estimate_ratio = st.sidebar.slider(
    "위험도 평가 밀집도 반영 비율",
    config.ESTIMATE_RATIO_MIN,
    config.ESTIMATE_RATIO_MAX,
    config.ESTIMATE_RATIO_DEFAULT,
    config.ESTIMATE_RATIO_STEP,
)

# 메인 메뉴
main_menu = st.radio(
    "Analysis mode",
    ["Density", "Flow"],
    horizontal=True,
    label_visibility="collapsed",
)


# 메인 - 밀집도 분석
if main_menu == "Density":
    uploaded_file = st.file_uploader(
        "Upload image (jpg, png)",
        type=["jpg", "jpeg", "png"],
        key="file_uploader",
    )

    view_col, data_col = st.columns([0.65, 0.35])

    with view_col:
        tab_bb, tab_heatmap, tab_raw = st.tabs(
            ["Bounding Box", "Density Heatmap", "Original"]
        )

        if uploaded_file:
            with st.spinner("Analyzing image..."):
                try:
                    res_bb, res_heatmap, count, risk_info = pipeline.process_all_views(
                        input_file=uploaded_file,
                        conf_threshold=conf_threshold,
                        gaussian_sigma=gaussian_sigma,
                        estimate_ratio= estimate_ratio
                    )

                    with data_col:
                        st.subheader("Results")
                        st.metric(label="감지 인원 수 : ", value=f"{count:03d}")

                        ## 추후 수정 필요
                        if risk_info["risk_level"] == "High":
                            st.error("Risk: high")
                        elif risk_info["risk_level"] == "Medium":
                            st.warning("Risk: medium")
                        else:
                            st.success("Risk: low")

                        st.metric(label="Risk Score", value=risk_info["risk_score"])
                        st.metric(label="최대 밀집도", value=risk_info["peak_density"])
                        st.metric(label="전체 공간 혼잡도", value=risk_info["occupancy_ratio"])
                         

                    with tab_bb:
                        st.image(
                            res_bb,
                            caption="bounding box result",
                            use_column_width=True,
                        )

                    with tab_heatmap:
                        caption = "Density Heatmap"
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

# 메인 - 흐름 분석
elif main_menu == "Flow":
    st.info("Flow estimation and tracking is planned.")
