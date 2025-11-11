import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import pandas as pd
from datetime import datetime
import argparse

# --- 1. 모델 및 데이터 처리 정의 ---

#  키 목록 정의 (test_clone_final.py의 JSON 출력 형식과 일치)
coord_keys = [
    "longi_left_upper_x", "longi_left_upper_y",
    "longi_left_lower_x", "longi_left_lower_y",
    "longi_right_upper_x", "longi_right_upper_y",
    "longi_right_lower_x", "longi_right_lower_y",
    "collar_left_lower_x", "collar_left_lower_y",
    "collar_left_upper_x", "collar_left_upper_y"
]
pose_keys = ["x", "y", "z", "pitch", "yaw", "roll"]

def extract_features(sample_dict):
    """JSON 딕셔너리에서 모델 입력 피처(29개)를 추출합니다."""
    f = [
        sample_dict["is_collar"],
        sample_dict["is_collar_hole"],
        sample_dict["collar_type"],
        sample_dict["is_hole_left"],
        sample_dict["is_hole_right"],
        sample_dict["hole_left_bounding_width"],
        sample_dict["hole_right_bounding_width"],
    ]
    f += [sample_dict["coordinates"][k] for k in coord_keys]
    f += [sample_dict["camera_pose"][k] for k in pose_keys]
    ps = sample_dict["pixel_scalar"]
    f += [ps["fillet"], ps["longi"], ps["collar_vertical"], ps["collar_horizontal"]]
    return np.array(f, dtype=np.float32)

def extract_labels(label_dict):
    """정답 라벨 JSON에서 실제 길이 값(4개)을 추출합니다."""
    return np.array([
        label_dict["fillet"],
        label_dict["longi"],
        label_dict["collar_vertical"],
        label_dict["collar_horizontal"]
    ], dtype=np.float32)

class CoordNet(nn.Module):
    """좌표 변환을 위한 MLP 모델 정의"""
    def __init__(self, input_dim=29, hidden1=256, hidden2=128, hidden3=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, output_dim),
        )
    def forward(self, x):
        return self.net(x)

# --- 2. 핵심 기능 함수 ---

def run_inference(model, device, input_dir, output_dir):
    """
    주어진 폴더의 모든 JSON에 대해 추론을 수행하고 결과를 별도 폴더에 저장합니다.
    """
    print(f"\n--- Running Inference ---")
    print(f"Input directory: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    sample_paths = glob.glob(os.path.join(input_dir, "*.json"))

    if not sample_paths:
        print(f"[ERROR] No JSON files found in {input_dir}. Make sure you run 'test_clone_final.py' first.")
        return

    for path in sample_paths:
        with open(path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        
        x_np = extract_features(sample)
        x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(x_tensor).cpu().numpy().squeeze()

        result = {
            'filename': os.path.basename(path),
            'fillet': float(y_pred[0]),
            'longi': float(y_pred[1]),
            'collar_vertical': float(y_pred[2]),
            'collar_horizontal': float(y_pred[3])
        }

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{base}.json")
        with open(out_path, 'w', encoding='utf-8') as out_f:
            json.dump(result, out_f, ensure_ascii=False, indent=4)
            
    print(f"Inference complete. {len(sample_paths)} results saved to: {output_dir}")

def run_evaluation(result_dir, labels_dir, outlier_threshold_cm=2.0):
    """
    추론 결과와 정답 라벨을 비교하여 성능 평가를 수행하고 CSV로 저장합니다.
    """
    print(f"\n--- Running Evaluation ---")
    print(f"Comparing results in '{result_dir}' with labels in '{labels_dir}'")

    label_paths = glob.glob(os.path.join(labels_dir, "*.json"))
    if not label_paths:
        print(f"[ERROR] No label files found in {labels_dir}. Cannot run evaluation.")
        return
        
    all_errors = []

    for label_path in label_paths:
        base = os.path.basename(label_path)
        result_path = os.path.join(result_dir, base)
        
        if not os.path.exists(result_path):
            print(f"[WARNING] Result file missing for label: {base}")
            continue

        with open(label_path, 'r', encoding='utf-8') as lf, \
             open(result_path, 'r', encoding='utf-8') as rf:
            label_data = json.load(lf)
            result_data = json.load(rf)
        
        if label_data.get('collar_vertical', 0) == 0:
            result_data['collar_vertical'] = 0
            result_data['collar_horizontal'] = 0

        file_errors = {'filename': base}
        for key in ['fillet', 'longi', 'collar_vertical', 'collar_horizontal']:
            if key in label_data and key in result_data:
                error = result_data[key] - label_data[key]
                file_errors[f'error_{key}'] = error
        all_errors.append(file_errors)

    if not all_errors:
        print("No matching files found to evaluate.")
        return

    df = pd.DataFrame(all_errors)
    summary_stats = {}
    for key in ['fillet', 'longi', 'collar_vertical', 'collar_horizontal']:
        error_col = f'error_{key}'
        valid_errors = df[error_col].dropna()
        
        summary_stats[key] = {
            'Mean_Error': valid_errors.mean(),
            'Mean_Absolute_Error(MAE)': valid_errors.abs().mean(),
            f'Outlier_Count(>{outlier_threshold_cm}cm)': (valid_errors.abs() > outlier_threshold_cm).sum()
        }

    summary_df = pd.DataFrame(summary_stats).T

    print("\n--- Performance Summary ---")
    print(summary_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv_path = os.path.join(result_dir, f"performance_evaluation_{timestamp}.csv")

    with open(output_csv_path, 'w', encoding='utf-8-sig') as f:
        f.write("--- Performance Summary ---\n")
        summary_df.to_csv(f)
        f.write("\n\n--- Detailed Errors ---\n")
        df.to_csv(f, index=False)

    print(f"\nEvaluation results saved to: {output_csv_path}")

# --- 3. 메인 실행 블록 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and evaluation on coordinate data.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the JSON files from test_clone_final.py (e.g., '.../final_json_results').")
    parser.add_argument("--model_path", default="coordinate_to_length_pt.pth", help="Path to the trained model (.pth file).")
    parser.add_argument("--labels_dir", default=None, help="(Optional) Directory containing the ground truth label JSON files for evaluation.")
    
    args = parser.parse_args()

    # --- 모델 및 디바이스 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CoordNet().to(device)
    try:
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found at {args.model_path}. Please check the path.")
        sys.exit(1)
        
    # --- 추론 실행 ---
    # 결과는 입력 폴더의 상위 폴더 밑 'inference_results'에 저장
    result_dir = os.path.join(os.path.dirname(args.input_dir.rstrip('/\\')), 'inference_results')
    run_inference(model, device, args.input_dir, result_dir)

    # --- 평가 실행 (라벨 폴더가 주어진 경우) ---
    if args.labels_dir:
        run_evaluation(result_dir, args.labels_dir)
    else:
        print("\n--labels_dir not provided, skipping evaluation.") 