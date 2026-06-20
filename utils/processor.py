import cv2
import numpy as np
from ultralytics import YOLO

from utils.density import DensityEstimator
from utils.heatmap import HeatmapGenerator
from utils.RiskEvaluator import RiskEvaluator


class VisionPipeline:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)

    def process_all_views(
        self,
        input_file,
        conf_threshold,
        gaussian_sigma,
        estimate_ratio,
    ):
        img_bytes = input_file.getvalue()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode the uploaded image.")

        #model inference
        results = self.model.predict(img, conf=conf_threshold, verbose=False)
        result = results[0]

        plotted_bgr = result.plot()
        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

        # BB
        bboxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else []
        
        # 밀집도 계산
        density_estimater = DensityEstimator(gaussian_sigma)
        density_map = density_estimater.estimate(image_shape=img.shape
                                                 , bboxes=bboxes )

        # 밀집도 히트맵 생성
        heatmap_gen = HeatmapGenerator(alpha=0.9)
        res_heatmap = heatmap_gen.render(image= img, density_map=density_map)
        res_heatmap = cv2.cvtColor(res_heatmap, cv2.COLOR_BGR2RGB) 

        # 위험도 평가
        risk_evaluator = RiskEvaluator(estimate_ratio=estimate_ratio)
        risk_info = risk_evaluator.evaluate(density_map=density_map)

        count = int(len(result.boxes))

        return plotted_rgb, res_heatmap, count, risk_info
