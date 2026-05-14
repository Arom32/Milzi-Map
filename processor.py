import cv2
import numpy as np
from ultralytics import YOLO

class VisionPipeline:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)

    def process_image(self, image_bytes, conf_threshold=0.25):
        # 1. Byte -> OpenCV 포맷(BGR) 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. YOLO 추론
        results = self.model.predict(img, conf=conf_threshold, verbose=False)
        
        # 3. 시각화 및 BGR -> RGB 변환 (UI 출력용)
        res_plotted = results[0].plot()
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # 4. 카운팅
        count = len(results[0].boxes)
        
        return res_plotted_rgb, count