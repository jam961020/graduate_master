"""
YOLO ROI 검출 모듈
"""
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class YOLODetector:
    """YOLO 기반 ROI 검출기"""
    
    CLASS_NAMES = [
        "fillet_hole",    # 0
        "fillet_WL",      # 1
        "longi_WL",       # 2
        "plate_behind",   # 3
        "plate_type0",    # 4
        "plate_type1",    # 5
        "plate_type2"     # 6
    ]
    
    def __init__(self, model_path="models/best.pt"):
        """
        Args:
            model_path: YOLO 모델 파일 경로
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO 모델을 찾을 수 없습니다: {model_path}")
        
        self.model = YOLO(str(self.model_path))
        print(f"[INFO] YOLO 모델 로드 완료: {model_path}")
    
    def detect_rois(self, image, conf_threshold=0.6):
        """
        이미지에서 ROI 검출
        
        Args:
            image: BGR 이미지
            conf_threshold: 신뢰도 임계값
        
        Returns:
            rois: list of (class_id, x1, y1, x2, y2)
        """
        results = self.model(image, verbose=False, conf=conf_threshold)
        
        rois = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                class_id = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                rois.append((class_id, x1, y1, x2, y2))
        
        # 클래스 ID 기준으로 정렬
        rois.sort(key=lambda x: x[0])
        
        return rois
    
    def get_roi_image(self, image, roi):
        """
        ROI 영역 이미지 추출
        
        Args:
            image: 원본 이미지
            roi: (class_id, x1, y1, x2, y2)
        
        Returns:
            roi_image: 잘라낸 이미지
            offset: (x1, y1) - 전역 좌표 변환용
        """
        _, x1, y1, x2, y2 = roi
        roi_image = image[y1:y2, x1:x2]
        return roi_image, (x1, y1)
    
    @staticmethod
    def roi_to_global(coords_roi, offset):
        """
        ROI 좌표를 전역 좌표로 변환
        
        Args:
            coords_roi: ROI 내 좌표 (x, y) or [(x1,y1), (x2,y2)]
            offset: (x_offset, y_offset)
        
        Returns:
            전역 좌표
        """
        x_off, y_off = offset
        
        if isinstance(coords_roi, tuple) and len(coords_roi) == 2:
            # 단일 점
            return (coords_roi[0] + x_off, coords_roi[1] + y_off)
        elif isinstance(coords_roi, list):
            # 선분 [(x1,y1), (x2,y2)]
            return [(x + x_off, y + y_off) for x, y in coords_roi]
        elif isinstance(coords_roi, np.ndarray) and coords_roi.shape == (4,):
            # [x1, y1, x2, y2]
            return np.array([
                coords_roi[0] + x_off,
                coords_roi[1] + y_off,
                coords_roi[2] + x_off,
                coords_roi[3] + y_off
            ])
        else:
            raise ValueError(f"지원하지 않는 좌표 형식: {type(coords_roi)}")


if __name__ == "__main__":
    # 테스트 코드
    import cv2
    
    # 모델 경로 설정 (실제 경로로 수정 필요)
    MODEL_PATH = "models/best.pt"
    
    try:
        detector = YOLODetector(MODEL_PATH)
        print("✓ YOLO 검출기 초기화 성공")
        
        # 테스트 이미지
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        rois = detector.detect_rois(test_img)
        print(f"✓ 검출된 ROI 개수: {len(rois)}")
        
    except FileNotFoundError as e:
        print(f"✗ 오류: {e}")
        print("\n다음 명령으로 모델을 models/ 폴더에 복사하세요:")
        print("  mkdir -p BO_optimization/models")
        print("  cp path/to/your/best.pt BO_optimization/models/")