import sys
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from heatmap import HeatmapGenerator

class VisionPipeline:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)
    
    def process_all_views(self, input_file, conf_threshold, grid_shape, gussian_sigma):
        # Byte -> OpenCV 포맷(BGR) 변환
        img_bytes = input_file.getvalue()
        org_img = Image.open(input_file)

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # YOLO 추론
        results = self.model.predict(img, conf=conf_threshold, verbose=False)
        
        # 시각화 및 BGR -> RGB 변환 (UI 출력용)
        res_plotted = results[0].plot()
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        bboxes = []
        # if len(results[0].boxes) > 0:
            # YOLO 결과에서 바운딩 박스 좌표 추출 (x1, y1, x2, y2 포맷)
        bboxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        print(
            f"Detected objects: {len(bboxes)} boxes\n"
            f"- Bounding Boxes (x1, y1, x2, y2): {bboxes}",
            flush=True,
        )
        sys.stdout.flush()
        

        heatmap_gen = HeatmapGenerator(
            image=img,
            bboxes=bboxes, 
            src_points=[(100, 100), (500, 100), (500, 400), (100, 400)], 
            grid_shape=grid_shape,
            sigma=gussian_sigma
        )
        # 밀집도 히트맵 생성 (
        res_heatmap = heatmap_gen.draw_density_heatmap()
        res_heatmap = cv2.cvtColor(res_heatmap, cv2.COLOR_BGR2RGB) 

        # 카운팅
        count = len(results[0].boxes)

        return res_plotted_rgb, res_heatmap, count