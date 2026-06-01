
PAGE_TITLE = "Milzi-Map"
PAGE_LAYOUT = "wide"
MAIN_TITLE = "밀지Map! ( Milzi-Map )"

DEFAULT_MODEL = "best.pt"
AVAILABLE_MODELS = ["best.pt", "yolov8n.pt"]

# Confidence 임계값
DEFAULT_CONF_THRESHOLD = 0.5
CONF_MIN = 0.0
CONF_MAX = 1.0
CONF_STEP = 0.05

# 그리드(Grid) 설정
DEFAULT_GRID_ROWS = 10
DEFAULT_GRID_COLS = 10
GRID_MIN = 5
GRID_MAX = 30

# 가우시안 블러 시그마 (Gaussian Blur Sigma)
DEFAULT_GAUSSIAN_SIGMA = 80.0
SIGMA_MIN = 0.0
SIGMA_MAX = 100.0
SIGMA_STEP = 5.0

# 임계치(Threshold) 및 알림 설정 
WARNING_THRESHOLD = 10
DANGER_THRESHOLD = 20