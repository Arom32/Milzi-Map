import cv2
import numpy as np
from ultralytics import YOLO

from utils.density import DensityEstimator
from utils.heatmap import HeatmapGenerator


class VisionPipeline:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)

    def process_all_views(
        self,
        input_file,
        conf_threshold,
        grid_shape,
        gaussian_sigma,
        axis_config=None,
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
        density_estimator = DensityEstimator(
            image_shape=img.shape,
            axis_config=axis_config,
            grid_shape=grid_shape,
            sigma=gaussian_sigma,
            
        )
        
        # 밀집도
        density_result = density_estimator.estimate(bboxes)

        # 히트맵
        heatmap_gen = HeatmapGenerator()
        heatmap_bgr = heatmap_gen.render(
            image=img,
            density_map=density_result.density_map,
            virtual_to_src_matrix=density_result.virtual_to_src_matrix,
        )
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        count = int(len(result.boxes))
        return plotted_rgb, heatmap_rgb, count, density_result
