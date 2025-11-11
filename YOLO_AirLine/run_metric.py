import json
import numpy as np
import os

def run_formula_based_inference(input_dir, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for fname in json_files:
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(result_dir, fname)

        with open(input_path, 'r') as f:
            data = json.load(f)

        coords = data['coordinates']

        # 좌표 기반 거리 계산
        collar1_scalar = np.linalg.norm([
            coords['longi_left_lower_x'] - coords['collar_left_lower_x'],
            coords['longi_left_lower_y'] - coords['collar_left_lower_y']
        ])

        collar2_scalar = np.linalg.norm([
            coords['longi_right_lower_x'] - coords['collar_left_lower_x'],
            coords['longi_right_lower_y'] - coords['collar_left_lower_y']
        ])

        collar_vertical = np.linalg.norm([
            coords['collar_left_lower_x'] - coords['collar_left_upper_x'],
            coords['collar_left_lower_y'] - coords['collar_left_upper_y']
        ])

        collar_horizontal = min(collar1_scalar, collar2_scalar)
        fillet_scalar = max(collar1_scalar, collar2_scalar)

        # collar가 없을 경우 처리
        if data['is_collar'] == 0:
            collar_horizontal = 0
            collar_vertical = 0
            fillet_scalar = np.linalg.norm([
                coords['longi_right_lower_x'] - coords['longi_left_lower_x'],
                coords['longi_right_lower_y'] - coords['longi_left_lower_y']
            ])

        # 최종 결과 생성
        result_data = {
            'file': fname,
            'fillet': fillet_scalar / 20.0,
            'longi': data["pixel_scalar"]["longi"] / 20.0,
            'collar_vertical': collar_vertical / 20.0,
            'collar_horizontal': collar_horizontal / 20.0,
        }

        # JSON 결과 저장
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)

        print(f"[INFO] {fname}에 대한 수식 기반 추론 결과 저장 완료")
