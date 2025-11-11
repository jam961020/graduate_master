import os
import json
import numpy as np
from pathlib import Path

def determine_hole_radius(bbox_width):
    ratio = bbox_width*0.75 / 20.0
    print(f"Ratio: {ratio}")
    if ratio < 6.25:
        return 5.0
    elif 6.25 <= ratio < 8.75:
        return 7.5
    elif 8.75 <= ratio < 11.25:
        return 10
    elif 11.25 <= ratio < 13.75:
        return 12.5
    else:
        return 15

def calculate_final_information(input_json_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    json_files = [f for f in os.listdir(input_json_dir) if f.endswith('.json')]

    results = []

    for fname in json_files:
        with open(os.path.join(input_json_dir, fname), 'r') as f:
            data = json.load(f)

        coords = data['coordinates']

        longi_left = np.linalg.norm([
            coords['longi_left_lower_x'] - coords['longi_left_upper_x'],
            coords['longi_left_lower_y'] - coords['longi_left_upper_y']
        ]) / 20.0 if coords['longi_left_upper_x'] != 0 else 0

        longi_right = np.linalg.norm([
            coords['longi_right_lower_x'] - coords['longi_right_upper_x'],
            coords['longi_right_lower_y'] - coords['longi_right_upper_y']
        ]) / 20.0 if coords['longi_right_upper_x'] != 0 else 0

        collar_vertical = np.linalg.norm([
            coords['collar_left_lower_x'] - coords['collar_left_upper_x'],
            coords['collar_left_lower_y'] - coords['collar_left_upper_y']
        ]) / 20.0 if coords['collar_left_upper_x'] != 0 else 0

        fillet_left = np.linalg.norm([
            coords['longi_left_lower_x'] - coords['collar_left_lower_x'],
            coords['longi_left_lower_y'] - coords['collar_left_lower_y']
        ]) / 20.0 if coords['collar_left_lower_x'] != 0 else 0

        fillet_right = np.linalg.norm([
            coords['longi_right_lower_x'] - coords['collar_left_lower_x'],
            coords['longi_right_lower_y'] - coords['collar_left_lower_y']
        ]) / 20.0 if coords['collar_left_lower_x'] != 0 else 0

        collar_horizontal = min(fillet_left, fillet_right) if data['is_collar'] else 0

        fillet_dist = np.linalg.norm([
            coords['longi_right_lower_x'] - coords['longi_left_lower_x'],
            coords['longi_right_lower_y'] - coords['longi_left_lower_y']
        ]) / 20.0

        result = {
            'longi_left': longi_left,
            'longi_right': longi_right,
            'longi_length': fillet_dist,
            'fillet_left': fillet_left,
            'fillet_right': fillet_right,
            'collar_vertical': collar_vertical,
            'collar_horizontal': collar_horizontal,
            'collar_type': data['collar_type'],
        }

        # collar 위치
        if data['is_collar']:
            result['is_collar_left'] = int(fillet_left <= fillet_right)
            result['is_collar_right'] = int(fillet_right < fillet_left)
        else:
            result['is_collar_left'] = 0
            result['is_collar_right'] = 0

        # hole 존재 여부 및 반경 계산 (AirLine JSON 호환)
        hole_left_w = float(data.get('hole_left_bounding_width', 0))
        hole_right_w = float(data.get('hole_right_bounding_width', 0))
        bbox_width = max(hole_left_w, hole_right_w)
        if bbox_width > 0:
            result['hole_radius'] = determine_hole_radius(bbox_width)
        else:
            # 구형 키를 사용하는 입력을 위한 폴백 (is_hole + hole_upper/lower 좌표)
            if data.get('is_hole', False) and all(k in coords for k in ['hole_lower_x','hole_upper_x','hole_lower_y','hole_upper_y']):
                bbox_width = np.linalg.norm([
                    coords['hole_lower_x'] - coords['hole_upper_x'],
                    coords['hole_lower_y'] - coords['hole_upper_y']
                ])
                result['hole_radius'] = determine_hole_radius(bbox_width)
            else:
                result['hole_radius'] = 0

        # 필렛 용접장 보정: determine_hole_radius 반환(cm)을 그대로 차감
        hole_radius_cm = float(result.get('hole_radius', 0))
        if hole_radius_cm > 0:
            is_left = int(data.get('is_hole_left', 0)) == 1
            is_right = int(data.get('is_hole_right', 0)) == 1
            side = 'left' if (is_left and not is_right) else ('right' if (is_right and not is_left) else None)
            if side is None:
                side = 'left' if hole_left_w >= hole_right_w else 'right'
            if side == 'left':
                before = result['fillet_left']
                result['fillet_left'] = max(0.0, result['fillet_left'] - hole_radius_cm)
                print(f"[SUB] side=left, hole_radius_cm={hole_radius_cm:.2f}, fillet_left: {before:.2f} -> {result['fillet_left']:.2f}")
            else:
                before = result['fillet_right']
                result['fillet_right'] = max(0.0, result['fillet_right'] - hole_radius_cm)
                print(f"[SUB] side=right, hole_radius_cm={hole_radius_cm:.2f}, fillet_right: {before:.2f} -> {result['fillet_right']:.2f}")

        results.append(result)

    # 평균 계산
    N = len(results)
    if N == 0:
        print("[경고] 처리할 파일이 없습니다.")
        return

    selected_results = results if N <= 10 else results[:10]

    mean_result = {
        key: np.mean([x[key] for x in selected_results]) for key in [
            'longi_left', 'longi_right', 'longi_length', 'fillet_left',
            'fillet_right', 'collar_vertical', 'collar_horizontal', 'hole_radius'
        ]
    }

    mean_result['collar_type'] = max(
        set([x['collar_type'] for x in selected_results]),
        key=[x['collar_type'] for x in selected_results].count
    )

    # Collar 방향 결정 (짧은 fillet이 있는 쪽이 Collar)
    mean_result['is_collar_left'] = int(mean_result['fillet_left'] <= mean_result['fillet_right'])
    mean_result['is_collar_right'] = int(mean_result['fillet_right'] < mean_result['fillet_left'])

    # 최종 JSON 저장
    output_path = os.path.join(output_dir, 'final_pendant_information.json')
    with open(output_path, 'w') as f:
        json.dump(mean_result, f, indent=4)

    print(f"[INFO] 최종 정보가 저장되었습니다: {output_path}")
