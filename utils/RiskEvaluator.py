import numpy as np
class RiskEvaluator:
    
    def __init__(self, 
                 peak_high=3.0,
                 occupancy_threshold=0.8, 
                 occ_high=0.3
                 , estimate_ratio = 0.5):
        """
        위험도 판단을 위한 하이퍼파라미터 초기화
        - peak_high : 절대 최대 밀집도(Peak Density) 기준 높다고 판단 할 수치 - 3명 정도가 밀집
        - occupancy_threshold: 공간 점유 픽셀로 간주할 최소 밀집도 수치
        - occ_high : 공간 점유율(Occupancy Ratio) 기준 (0.0 ~ 1.0)
        """
        self.peak_high = peak_high
        self.occupancy_threshold = occupancy_threshold
        self.occ_high = occ_high
        self.estimate_ratio = estimate_ratio

    def evaluate(self, density_map: np.ndarray) -> dict:
        """
        2D 밀집도 맵을 분석하여 위험도 지표 반환
        """
        if density_map.size == 0:
            return {"risk_level": "Low", "peak_density": 0.0, "occupancy_ratio": 0.0, "risk_score": 0.0}

        # 절대적 임계값 산출
        peak_density = float(density_map.max())

        # 공간 점유율 산출
        crowded_pixels = np.sum(density_map >= self.occupancy_threshold)
        occupancy_ratio = float(crowded_pixels / density_map.size)

        # 종합 위험도 점수 산출
        # 각 지표를 High 기준치로 정규화
        norm_peak = peak_density / self.peak_high
        norm_occ = occupancy_ratio / self.occ_high
        
        # 가중치 합산 연산 (Weighted Linear Combination)
        risk_score = (norm_peak * self.estimate_ratio) + (norm_occ * (1.0 - self.estimate_ratio))

        # 3. 하이브리드 위험도 (Hybrid Risk Level) 판별
        if risk_score >= 1.0:
            risk_level = "High"
        elif risk_score >= 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "risk_level": risk_level,
            "peak_density": peak_density,
            "occupancy_ratio": occupancy_ratio,
            "risk_score": round(risk_score, 4)
        }