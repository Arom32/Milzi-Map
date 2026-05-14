import streamlit as st
from processor import VisionPipeline

# 1. 초기 설정 및 파이프라인 캐싱
st.set_page_config(page_title="YOLOv8 Image Detector", layout="wide")

if 'pipeline' not in st.session_state:
    st.session_state.pipeline = VisionPipeline(model_path="best.pt")

pipeline = st.session_state.pipeline

st.title("YOLOv8 단일 이미지 객체 검출")

# 2. 사이드바 설정
st.sidebar.header("설정")
conf_threshold = st.sidebar.slider("Confidence 임계값", 0.0, 1.0, 0.5, 0.05)

# 3. 메인 레이아웃 (좌: 원본, 우: 결과)
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 원본 이미지")
    uploaded_file = st.file_uploader("이미지 업로드 (jpg, png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="업로드된 이미지", use_container_width=True)

with col2:
    st.subheader("2. 검출 결과")
    # 버튼 로직을 제거하고 업로드 즉시 실행되도록 변경
    if uploaded_file:
        with st.spinner("객체를 검출하는 중입니다..."):
            try:
                # 파일 바이트 추출 및 파이프라인 전달
                img_bytes = uploaded_file.getvalue()
                res_img, count = pipeline.process_image(img_bytes, conf_threshold)
                
                # 결과 렌더링
                st.image(res_img, caption="YOLOv8 검출 결과", use_container_width=True)
                st.success("검출 완료!")
                st.metric(label="탐지된 객체 수", value=f"{count} 개")
            
            except Exception as e:
                st.error(f"오류 발생: {e}")
    else:
        st.info("이미지를 업로드하면 검출이 시작됩니다.")