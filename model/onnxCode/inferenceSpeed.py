
import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
from ultralytics import YOLO  # YOLO11 .pt 로드를 위해 필요

# ==========================================
# 1. PyTorch (.pt) 추론 속도 측정 함수
# ==========================================
def run_pytorch_inference(model_path, input_shape, iterations=50):
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # YOLO 모델 로드 및 순수 PyTorch 모델 객체 추출
    try:
        model = YOLO(model_path)
        pt_model = model.model.to(device)
        pt_model.eval()
    except Exception as e:
        print(f"  -> 모델 로드 실패: {e}")
        return
    
    # 더미 입력 텐서 생성
    dummy_input = torch.randn(input_shape).to(device)
    
    # 웜업 (Warm-up)
    with torch.no_grad():
        for _ in range(10):
            _ = pt_model(dummy_input)
            
    # 본 측정
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = pt_model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize() # GPU 연산 완료 대기 (필수)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            
    avg_latency = np.mean(latencies)
    print(f"  -> 평균 추론 시간: {avg_latency:.2f} ms")
    return avg_latency

# ==========================================
# 2. ONNX (.onnx) 추론 속도 측정 함수
# ==========================================
def run_onnx_inference(model_path, input_shape, iterations=50):
    # Execution Provider 설정 (CUDA 우선)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as e:
        print(f"  -> 모델 로드 실패: {e}")
        return
        
    input_name = session.get_inputs()[0].name
    
    # 더미 입력 텐서 생성
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 웜업 (Warm-up)
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
        
    # 본 측정
    latencies = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
        
    avg_latency = np.mean(latencies)
    print(f"  -> 평균 추론 시간: {avg_latency:.2f} ms")
    return avg_latency

# ==========================================
# 3. 폴더 재귀 탐색 및 자동 텐서 할당 메인 로직
# ==========================================
def benchmark_folder(folder_path, iterations=100):
    target_dir = Path(folder_path)
    
    if not target_dir.exists():
        print(f"경로를 찾을 수 없습니다: {folder_path}")
        return
        
    # 1. rglob을 사용하여 하위 폴더를 포함해 'best.pt'와 'best.onnx'를 모두 찾음
    pt_files = list(target_dir.rglob("best.pt"))
    onnx_files = list(target_dir.rglob("best.onnx"))
    
    print(f"=== 벤치마크 시작 (총 .pt: {len(pt_files)}개, .onnx: {len(onnx_files)}개) ===\n")
    
    # 2. PyTorch 모델 벤치마크
    for pt_file in pt_files:
        model_name = pt_file.parent.parent.name
        # 이름에 '640'이 들어가 있으면 640 텐서, 아니면 960 텐서 자동 할당
        current_shape = (1, 3, 640, 640) if "640" in model_name else (1, 3, 960, 960)
        
        print(f"[PyTorch] Testing: {model_name} (Shape: {current_shape[2]}x{current_shape[3]})")
        run_pytorch_inference(pt_file, current_shape, iterations)
        print("-" * 50)
        
    # 3. ONNX 모델 벤치마크
    for onnx_file in onnx_files:
        model_name = onnx_file.parent.parent.name
        current_shape = (1, 3, 640, 640) if "640" in model_name else (1, 3, 960, 960)
        
        print(f"[ONNX] Testing: {model_name} (Shape: {current_shape[2]}x{current_shape[3]})")
        run_onnx_inference(onnx_file, current_shape, iterations)
        print("-" * 50)

    print("\n=== 벤치마크 완료 ===")

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    # 최상위 모델 폴더 경로 지정
    TARGET_FOLDER = "/workspace/code/onnx1" 
    
    # 100회 반복 측정 실행
    benchmark_folder(TARGET_FOLDER, iterations=100)
