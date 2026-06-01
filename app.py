import streamlit as st
from PIL import Image
from processor import VisionPipeline

# 1. 초기 설정 및 파이프라인 캐싱
st.set_page_config(page_title="Milzi-Map", layout="wide")


selected_model = "best.pt" 

if 'pipeline' not in st.session_state or st.session_state.get('current_model') != selected_model:
    st.session_state.pipeline = VisionPipeline(model_path=selected_model)
    st.session_state.current_model = selected_model


pipeline = st.session_state.pipeline

st.title("밀지Map! ( Milzi-Map )")

#사이드바 설정
st.sidebar.header("설정")
conf_threshold = st.sidebar.slider("Confidence 임계값", 0.0, 1.0, 0.5, 0.05)

grid_rows = st.sidebar.slider("그리드 행(Rows) 개수", 5, 30, 10)
grid_cols = st.sidebar.slider("그리드 열(Cols) 개수", 5, 30, 10)
gussian_sigma = st.sidebar.slider("가우시안 블러 시그마", 0.0, 100.0, 80.0, 5.0)
grid_shape = (grid_rows, grid_cols)

st.sidebar.header("YOLOv8 모델 설정")

# 모델 선택 (n, s, m 등 사이즈별 비교 유도)
selected_model = st.sidebar.selectbox("Detection 모델을 선택하세요.", 
                                  ["best.pt","yolov8n.pt"])


if 'pipeline' not in st.session_state or st.session_state.get('current_model') != selected_model:
    with st.spinner(f"모델 '{selected_model}'을(를) 로딩 중입니다..."):
        st.session_state.pipeline = VisionPipeline(model_path=selected_model)
        st.session_state.current_model = selected_model

# 현재 활성화된 파이프라인 객체 바인딩
pipeline = st.session_state.pipeline

# 큰 메뉴: 밀집도 분석 / 방향성 분석
main_menu = st.radio(
    "분석 모드 선택", 
    ["밀집도 분석", "이동 방향성 분석"], 
    horizontal=True,
    label_visibility="collapsed"
)

# 메인 인터페이스
if main_menu == "밀집도 분석":
    
    # 렌더링 병목을 방지하기 위해 컬럼 스플릿 전에 파일 업로더 변수를 전역 스코프 수준에서 선언
    # 이를 통해 데이터 컨트롤러와 시각화 탭이 동시에 동일 오브젝트를 완전하게 참조 가능
    uploaded_file = st.file_uploader(
        "이미지 업로드 (jpg, png)", 
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )
    
    st.markdown("---")

    # 영역 분할 (좌측 65% 시각화 패널, 우측 35%: 데이터 및 컨트롤 패널)
    view_col, data_col = st.columns([0.65, 0.35])
    
    with data_col:
        # st.subheader("1. 시스템 제어 및 상태")
        # 텍스트 레이아웃 배치 공간 확보
        status_box = st.empty() 
        
    with view_col:
        # 큰 옵션 내부의 작은 옵션 3가지 (서브 탭 구현)
        tab_bb, tab_heatmap, tab_raw = st.tabs(["Bounding Box 출력", "Heatmap 출력", "원본 이미지"])
        
        if uploaded_file:           
            with st.spinner("이미지를 분석 중입니다..."):
                try:
                    # 백엔드 파이프라인에서 이미지 일괄 추출
                    res_bb, res_heatmap, count = pipeline.process_all_views(
                        input_file=uploaded_file,
                        conf_threshold=conf_threshold,
                        grid_shape=grid_shape,
                        gussian_sigma=gussian_sigma
                    )
                    
                    # 데이터 패널 수치 동적 업데이트 반영
                    with data_col:
                        st.subheader("실시간 분석 결과")
                        st.metric(label="검출된 총 인원수", value=f"{count:03d} 명")
                        
                        # 인원수에 따른 위험도 정량화 바인딩
                        if count >= 20:
                            st.error("위험도: 높음 ")
                        elif count >= 10:
                            st.warning("위험도: 보통 ")
                        else:
                            st.success("위험도: 낮음 ")
                    
                    # 탭별 이미지 스케일링 바인딩
                    with tab_bb:
                        st.image(res_bb, caption="YOLOv8 Bounding Box 탐지 결과", width='stretch')
                    
                    with tab_heatmap:
                        st.image(res_heatmap, caption="타일 기반 왜곡 보정 히트맵 결과", width='stretch')
                        
                    with tab_raw:
                        st.image(Image.open(uploaded_file), caption="업로드 원본 데이터", width='stretch')
                        
                except Exception as e:
                    st.error(f"연산 처리 중 예외 발생: {e}")
        else:
            # 파일 미선택 시 플레이스홀더 바인딩
            with tab_bb: st.info("이미지를 업로드하면 Bounding Box 결과가 표출됩니다.")
            with tab_heatmap: st.info("이미지를 업로드하면 히트맵 연산 결과가 표출됩니다.")
            with tab_raw: st.info("이미지를 업로드하면 원본 프레임이 표출됩니다.")

elif main_menu == "이동 방향성 분석":
    st.info("이동 방향성 분석(Flow Estimation & Tracking) 모듈은 현재 준비 중입니다.")

