# Milzi-Map

Milzi-Map is a Streamlit application for visualizing crowd density from image
data. It detects people with YOLO, estimates each person's ground position,
corrects floor perspective from user-provided direction lines, and overlays a
density heatmap on the original image.

## Main Features

- YOLOv8 based object detection.
- Optional floor perspective correction from two X-direction lines and two
  Y-direction lines.
- Bottom-center point extraction from each bounding box.
- Local-plane grid counting and lightweight DBSCAN-style clustering.
- Gaussian KDE-style density accumulation.
- Heatmap rendering as a separate final overlay step.
- Risk level display based on the detected person count.
- Placeholder for future flow estimation and tracking.

## Environment

- Python 3.x
- Streamlit
- Ultralytics YOLO
- OpenCV
- NumPy
- Pillow
- streamlit-drawable-canvas

## Installation

```bash
git clone <repository-url>
cd Milzi-Map
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Density Pipeline

1. The user uploads an image through the Streamlit UI.
2. The user can draw two red X-direction floor lines and two blue Y-direction
   floor lines. Their intersections form the floor quadrilateral used for
   perspective correction. If lines are not provided, the image coordinate system
   is used as-is.
3. OpenCV decodes the uploaded bytes into a BGR image.
4. YOLO predicts bounding boxes for people.
5. Each bounding box is converted to a bottom-center point. This is more logical
   than the box center for crowd density because it approximates where the person
   stands on the ground plane.
6. `DensityEstimator` projects those points into the rectified floor plane.
7. Valid local-plane points are counted per grid cell and clustered with a small
   NumPy DBSCAN implementation.
8. Gaussian kernels are accumulated in the local plane to create a smooth density
   map.
9. `DensityEstimator` returns the density map and analysis metrics.
10. `HeatmapGenerator` only renders that density map: it warps the map back to
    the original image, applies `cv2.applyColorMap`, and alpha-blends the overlay.

## Code Responsibilities

- `app.py`: Streamlit UI, model/settings controls, floor-line input, result display.
- `axis_input.py`: image-canvas floor-line drawing and perspective config extraction.
- `processor.py`: orchestration between YOLO, density estimation, and rendering.
- `density.py`: coordinate correction, grid counting, DBSCAN, and density map
  generation.
- `heatmap.py`: heatmap rendering only.

## Notes

- The previous fixed ROI placeholder and manual four-corner polygon input were
  removed. Users now draw two X-direction and two Y-direction floor lines.
- `HeatmapGenerator` is intentionally render-only. Density math lives in
  `DensityEstimator`.
- DBSCAN is implemented locally to avoid adding a new runtime dependency.

# 밀지 맵 (Milzi-Map)

밀지 맵(Milzi-Map)은 이미지 데이터로부터 군중 밀집도를 시각화하는 Streamlit 애플리케이션입니다. YOLO를 사용하여 사람을 탐지하고, 각 사람의 지면 위치를 추정한 후, 사용자가 제공한 방향 선을 바탕으로 바닥 원근을 보정하여 원본 이미지 위에 밀집도 히트맵을 오버레이로 표시합니다.

## 주요 기능

* YOLOv8 기반 객체 탐지.
* 사용자가 그린 두 개의 X축 방향 선과 두 개의 Y축 방향 선을 이용한 바닥 원근 보정 (선택 사항).
* 각 바운딩 박스의 하단 중앙 점 추출.
* 로컬 평면(Local-plane) 그리드 카운팅 및 경량화된 DBSCAN 스타일 클러스터링.
* 가우시안 KDE 스타일 밀집도 누적.
* 별도의 최종 오버레이 단계로 히트맵 렌더링.
* 탐지된 인원수에 따른 위험도 레벨 표시.
* 향후 이동 방향성 분석(Flow Estimation) 및 트래킹 모듈 추가를 위한 자리 표시자(Placeholder) 포함.

## 개발 환경

* Python 3.x
* Streamlit
* Ultralytics YOLO
* OpenCV
* NumPy
* Pillow
* streamlit-drawable-canvas

## 설치 방법

```bash
git clone <repository-url>
cd Milzi-Map
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

```

## 실행 방법

```bash
streamlit run app.py

```

## 밀집도 분석 파이프라인 (Density Pipeline)

1. 사용자가 Streamlit UI를 통해 이미지를 업로드합니다.
2. 사용자는 두 개의 빨간색 X축 방향 선과 두 개의 파란색 Y축 방향 선을 그릴 수 있습니다. 이 선들의 교차점이 원근 보정에 사용되는 바닥 사각형(Quadrilateral)을 형성합니다. 선을 그리지 않을 경우 원본 이미지 좌표계가 그대로 사용됩니다.
3. OpenCV가 업로드된 바이트를 BGR 이미지로 디코딩합니다.
4. YOLO가 사람의 바운딩 박스를 예측합니다.
5. 각 바운딩 박스는 하단 중앙 점으로 변환됩니다. 이는 박스의 중앙보다 보행자가 실제 지면에 서 있는 위치에 가깝기 때문에 군중 밀집도 파악에 더 논리적입니다.
6. `DensityEstimator`가 해당 점들을 보정된 바닥 평면에 투영합니다.
7. 유효한 로컬 평면 점들은 그리드 셀 단위로 집계되며, NumPy로 구현된 소형 DBSCAN을 통해 클러스터링됩니다.
8. 가우시안 커널(Gaussian kernels)이 로컬 평면에 누적되어 부드러운 밀집도 맵(Density map)을 생성합니다.
9. `DensityEstimator`는 밀집도 맵과 분석 지표를 반환합니다.
10. `HeatmapGenerator`는 반환된 맵을 렌더링하는 역할만 수행합니다: 맵을 원본 이미지로 다시 워핑(warping)하고, `cv2.applyColorMap`을 적용한 뒤, 알파 블렌딩(Alpha-blending)을 통해 오버레이합니다.

## 코드 구조 (Code Responsibilities)

* `app.py`: Streamlit UI, 모델/설정 제어, 바닥 선 입력 캔버스, 결과 화면 출력.
* `axis_input.py`: 이미지 캔버스 상의 바닥 선 그리기 및 원근 보정 설정 추출.
* `processor.py`: YOLO, 밀집도 추정, 렌더링 간의 오케스트레이션(Orchestration).
* `density.py`: 좌표 보정, 그리드 카운팅, DBSCAN 클러스터링, 밀집도 맵 생성.
* `heatmap.py`: 히트맵 렌더링 전용 모듈.

## 참고 사항

* 기존의 고정된 ROI 영역 및 수동 네 모서리(four-corner) 다각형 입력 방식은 제거되었습니다. 이제 사용자는 X축 방향 선 2개, Y축 방향 선 2개를 직접 그려서 설정합니다.
* `HeatmapGenerator`는 의도적으로 렌더링 전용으로 분리되었습니다. 밀집도 연산 로직은 `DensityEstimator`에 존재합니다.
* 새로운 런타임 종속성을 추가하지 않기 위해 DBSCAN은 스크립트 내부(로컬)에서 직접 구현되었습니다.