PAGE_TITLE = "밀지Map! (Milzi-Map!)"
PAGE_LAYOUT = "wide"
MAIN_TITLE = "밀지Map! (Milzi-Map!)"

DEFAULT_MODEL = "11n_base_640.onnx"
AVAILABLE_MODELS = ["11n_Base_960.onnx",
                    "8m_Base_960.onnx",
                    "8n_Base_960.onnx",
                    # "11n_base_960.pt",
                    "11n_base_640.onnx",
                    "11n_ConservativeT_960.onnx",
                    "11n_HatPlus_960.onnx",
                    "11n_PramOnly_960.onnx",
                    "11n_Tuned_960.onnx",
                    "26n_Base_960.onnx",
                    "yolov8n.pt",
                    ]

DEFAULT_CONF_THRESHOLD = 0.5
CONF_MIN = 0.0
CONF_MAX = 1.0
CONF_STEP = 0.05

DEFAULT_GRID_ROWS = 10
DEFAULT_GRID_COLS = 10
GRID_MIN = 5
GRID_MAX = 30

DEFAULT_GAUSSIAN_SIGMA = 50.0
SIGMA_MIN = 0.0
SIGMA_MAX = 100.0
SIGMA_STEP = 5.0

ESTIMATE_RATIO_MAX = 1.0
ESTIMATE_RATIO_MIN = 0.0
ESTIMATE_RATIO_DEFAULT = 0.5
ESTIMATE_RATIO_STEP = 0.1