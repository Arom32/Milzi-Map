from pathlib import Path
from ultralytics import YOLO

# # 1. 최상위 부모 폴더 경로 설정 (실제 환경에 맞게 수정)
# base_dir = Path('./')

# # 2. glob을 활용하여 모든 하위 폴더 내의 'weights/best.pt' 파일 일괄 탐색
# # '*/weights/best.pt'는 부모 폴더 바로 아래의 1 depth 하위 폴더들을 모두 검사합니다.
# pt_files = list(base_dir.glob('*/weights/best.pt'))

# print(f"총 {len(pt_files)}개의 best.pt 모델을 발견했습니다.\n{'='*40}")

# # 3. 자동 순회 및 변환 실행
# for pt_path in pt_files:
#     # 경로에서 모델 이름(폴더명) 추출 (예: n11_Base_960px)
#     model_name = pt_path.parent.parent.name
    
#     print(f"[{model_name}] ONNX 변환 시작...")
    
#     try:
#         # 모델 로드
#         model = YOLO(str(pt_path))
        
#         # ONNX 변환 (이전 최적화 옵션 및 GPU 배포용 half=True 적용)
#         onnx_path = model.export(
#             format='onnx',
#             imgsz=960,
#             simplify=True,
#             dynamic=False,
#             half=True  # FP16 양자화 (노트북 GPU 배포 시 권장)
#         )
#         print(f"  -> 변환 성공: {onnx_path}\n")
        
#     except Exception as e:
#         # 특정 모델 변환 실패 시 프로그램이 종료되지 않도록 예외 처리
#         print(f"  -> 변환 실패 ({model_name}): {e}\n")

# print(f"{'='*40}\n모든 일괄 변환 작업이 종료되었습니다.")


model = YOLO('/workspace/code/onnx1/11n_base_640/weights/best.pt')

# 2. ONNX 포맷으로 변환
path = model.export(
    format='onnx',      # 변환 타겟 포맷
    imgsz=640,          # 학습 시 사용한 이미지 크기 (고정)
    simplify=True,      # ONNX 그래프 구조 단순화 및 최적화 (권장)
    dynamic=False       # 동적 배치/이미지 크기 사용 여부 (False 권장: 속도 향상 및 호환성 확보)
)

print(f"ONNX 변환 완료. 저장 경로: {path}")