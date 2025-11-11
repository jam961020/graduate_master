import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import CRG311 as crg
from deximodel import DexiNed
import torch.nn as nn
import json

img_count = 0
# YOLO 모델 불러오기 및 검출 대상 이미지 폴더 경로 지정
model = YOLO(r".\best_l.pt")
img_folder = r"C:\Users\user\Desktop\study\task\weld2025\weld2025_samsung_git_temp\testing\samsung2024\realDataset_1\1_1"

img_files = [file for file in os.listdir(img_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

# 결과 저장 폴더 경로 설정
save_folder = os.path.join(os.getcwd(), r".\테스트_왜 안되는가")
os.makedirs(save_folder, exist_ok=True)

# Pendant_translator를 위한 json 파일 저장 경로 설정
save_folder_for_all_progoram = os.path.join(os.getcwd(), r".\coordinates_inference")
os.makedirs(save_folder_for_all_progoram, exist_ok=True)

THETARESOLUTION = 6
KERNEL_SIZE = 9

def buildOrientationDetector():
    thetaN = nn.Conv2d(1, THETARESOLUTION, KERNEL_SIZE, 1, KERNEL_SIZE // 2, bias=False).cuda()
    for i in range(THETARESOLUTION):
        kernel = np.zeros((KERNEL_SIZE, KERNEL_SIZE))
        angle = i * 180 / THETARESOLUTION
        x = (np.cos(angle / 180 * np.pi) * (KERNEL_SIZE // 2)).astype(np.int32)
        y = (np.sin(angle / 180 * np.pi) * (KERNEL_SIZE // 2)).astype(np.int32)
        
        cv2.line(kernel, (KERNEL_SIZE // 2 - x, KERNEL_SIZE // 2 - y), 
                 (KERNEL_SIZE // 2 + x, KERNEL_SIZE // 2 + y), 1, 1)
        thetaN.weight.data[i] = torch.tensor(kernel, dtype=torch.float32)
    return thetaN

OrientationDetector = buildOrientationDetector()

# Edge Detector 모델 (dexi.pth)
edgeDetector = DexiNed().cuda()
edge_state_dict = torch.load(r".\YOLO_AirLine\dexi.pth", map_location='cuda:0')
edgeDetector.load_state_dict(edge_state_dict)

usingUnet = 0
config = {
    "edgeThresh": 0,
    "simThresh": 0.9,
    "pixelNumThresh": 100,
}

tempMem = np.zeros((50000, 2), dtype=np.int32)
tempMem2 = np.zeros((2, 300000, 2), dtype=np.int32)
tempMem3 = np.zeros((3000, 2, 2), dtype=np.float32)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def get_longest_line(lines):
    max_length = 0
    longest_line = None
    for pt1, pt2 in lines:
        length = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        if length > max_length:
            max_length = length
            longest_line = (pt1, pt2)
    return longest_line

def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
    if m1 == m2:
        return None
    b1 = y1 - m1 * x1
    b2 = y3 - m2 * x3
    if m1 != float('inf') and m2 != float('inf'):
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    elif m1 == float('inf'):
        x = x1
        y = m2 * x + b2
    else:
        x = x3
        y = m1 * x + b1
    return (x, y)

# Coordinate Detector 검증을 위한 Ground truth 경로 지정 (필요없는 부분)
# ground_truth_folder = r".\Simulator Output\CoordinatesAnswer"

left_collar_count = 0
right_collar_count = 0
left_collar_sum = 0
right_collar_sum = 0

for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {img_path}")
        continue

    img_name = os.path.splitext(os.path.basename(img_path))[0]

    result = model(img, conf=0.3, iou=0.3)
    boxes = result[0].boxes

    airline_only_img = img.copy()
    yolo_only_img = img.copy()
    filtered_img = img.copy()

    rx1 = img.copy()

    original_height, original_width = rx1.shape[:2]

    if not usingUnet:
        res = 16
        dscale = 1
        resized_height = rx1.shape[0] // dscale // res * res
        resized_width = rx1.shape[1] // dscale // res * res
        rx1_resized = cv2.resize(rx1, (resized_width, resized_height))
    else:
        rx1_resized = rx1

    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    if len(rx1_resized.shape) == 2:
        rx1_resized = cv2.cvtColor(rx1_resized, cv2.COLOR_GRAY2RGB)
    elif rx1_resized.shape[2] == 4:
        rx1_resized = cv2.cvtColor(rx1_resized, cv2.COLOR_RGBA2RGB)

    rx1_resized = np.ascontiguousarray(rx1_resized)
    x1 = rx1_resized

    if usingUnet:
        x1 = cv2.cvtColor(x1, cv2.COLOR_RGB2GRAY)

    x1 = torch.tensor(x1, dtype=torch.float32).cuda() / 255.0

    if usingUnet:
        x1 = x1.unsqueeze(0)
    else:
        x1 = x1.permute(2, 0, 1)

    with torch.no_grad():
        edgeDetection = edgeDetector(x1.unsqueeze(0))
    ODes = OrientationDetector(edgeDetection)
    ODes = torch.nn.functional.normalize(ODes - ODes.mean(1, keepdim=True), p=2.0, dim=1)

    edgeNp = edgeDetection.detach().cpu().numpy()[0, 0]
    print(f"[DEBUG] Original Coordinate Detector - Edge 값 범위: {edgeNp.min():.6f} ~ {edgeNp.max():.6f}")
    
    outMap = np.zeros_like(edgeNp, dtype=np.uint8)
    outMap = np.expand_dims(outMap, 2).repeat(3, axis=2)
    out = np.zeros((3000, 2, 3), dtype=np.float32)

    edgeNp = (edgeNp > config["edgeThresh"]).astype(np.uint8) * 255
    print(f"[DEBUG] Original Coordinate Detector - Binary edge 픽셀 수: {np.sum(edgeNp > 0)}")

    rawLineNum = crg.desGrow(
        outMap, edgeNp, ODes[0].detach().cpu().numpy(), out,
        config["simThresh"], config["pixelNumThresh"],
        tempMem, tempMem2, tempMem3, THETARESOLUTION
    )
    
    print(f"[DEBUG] Original Coordinate Detector - 검출된 선 개수: {rawLineNum}")

    for i in range(rawLineNum):
        pt1 = (out[i, 0, 1] * scale_x, out[i, 0, 0] * scale_y)
        pt2 = (out[i, 1, 1] * scale_x, out[i, 1, 0] * scale_y)
        pt1 = (int(np.clip(pt1[0], 0, airline_only_img.shape[1] - 1)), int(np.clip(pt1[1], 0, airline_only_img.shape[0] - 1)))
        pt2 = (int(np.clip(pt2[0], 0, airline_only_img.shape[1] - 1)), int(np.clip(pt2[1], 0, airline_only_img.shape[0] - 1)))
        cv2.line(airline_only_img, pt1, pt2, (0, 255, 0), 2)

    annotator = Annotator(yolo_only_img)
    for box in boxes:
        b = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        conf = box.conf[0].cpu().numpy()
        label = f"{model.names[cls]} {conf:.2f}"
        # 바운딩 박스 색상을 빨간색으로 변경
        annotator.box_label(b, label, color=(163, 103, 0))
    yolo_only_img = annotator.result()

    is_collar = 0
    collar_type = 0
    is_hole_left = 0
    is_hole_right = 0
    hole_left_bounding_width = 0
    hole_right_bounding_width = 0

    coordinates = {
        "longi_left_upper_x": 0,
        "longi_left_upper_y": 0,
        "longi_left_lower_x": 0,
        "longi_left_lower_y": 0,
        "longi_right_upper_x": 0,
        "longi_right_upper_y": 0,
        "longi_right_lower_x": 0,
        "longi_right_lower_y": 0,
        "collar_lower_x": 0,
        "collar_lower_y": 0,
        "collar_upper_x": 0,
        "collar_upper_y": 0
    }

    longest_cls1_line = None
    longest_cls2_left_line = None
    longest_cls2_right_line = None
    found_cls1_line = False
    found_cls2_left_line = False
    found_cls2_right_line = False

    collar_left_flag = False
    for box in boxes:
        b = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        b = b.astype(np.float32)

        x_min, y_min, x_max, y_max = b
        bbox = [x_min, y_min, x_max, y_max]

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        if cls in [3, 4, 5, 6]:
            is_collar = 1
            collar_type_mapping = {3: 0, 4: 1, 5: 2, 6: 3}
            collar_type = collar_type_mapping[cls]
            
            if (x_min < 640):
                left_collar_count += 1
                coordinates["collar_upper_x"] = x_max-7
                coordinates["collar_upper_y"] = y_min+15
                coordinates["collar_lower_x"] = x_max-7
                coordinates["collar_lower_y"] = y_max-12
            else:
                right_collar_count += 1
                coordinates["collar_upper_x"] = x_min+5
                coordinates["collar_upper_y"] = y_min+18
                coordinates["collar_lower_x"] = x_min+5
                coordinates["collar_lower_y"] = y_max-12
            
            cv2.line(filtered_img, (int(coordinates["collar_upper_x"]), int(coordinates["collar_upper_y"])), (int(coordinates["collar_lower_x"]), int(coordinates["collar_lower_y"])), (0, 0, 255), 2)
            pass
        elif cls == 0:
            if center_x < img.shape[1] / 2:
                is_hole_left = 1
                hole_left_bounding_width = x_max - x_min - 15
            else:
                is_hole_right = 1
                hole_right_bounding_width = x_max - x_min - 15
        elif cls == 1:
            lines_in_box = []
            for i in range(rawLineNum):
                pt1 = (out[i, 0, 1] * scale_x, out[i, 0, 0] * scale_y)
                pt2 = (out[i, 1, 1] * scale_x, out[i, 1, 0] * scale_y)

                mid_pt = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

                if (x_min <= mid_pt[0] <= x_max) and (y_min <= mid_pt[1] <= y_max):
                    delta_x = pt2[0] - pt1[0]
                    delta_y = pt2[1] - pt1[1]
                    if delta_x == 0:
                        delta_x = 1e-6
                    if abs(delta_y / delta_x) < np.tan(np.deg2rad(30)):
                        lines_in_box.append((pt1, pt2))

            longest_line = get_longest_line(lines_in_box)
            if longest_line:
                pt1, pt2 = longest_line
                longest_cls1_line = (pt1, pt2)
                found_cls1_line = True

                pt1 = np.array(pt1)
                pt2 = np.array(pt2)
                pt1[0] = np.clip(pt1[0], x_min, x_max)
                pt1[1] = np.clip(pt1[1], y_min, y_max)
                pt2[0] = np.clip(pt2[0], x_min, x_max)
                pt2[1] = np.clip(pt2[1], y_min, y_max)

                bbox_width = x_max - x_min - 30
                required_length = 0.95 * bbox_width
                line_length = abs(pt2[0] - pt1[0])

                if line_length < required_length:
                    y = (pt1[1] + pt2[1]) / 2

                    x1 = x_min + 15
                    x2 = x_max - 15
                    pt1 = np.array([x1, y])
                    pt2 = np.array([x2, y])

                pt1 = (int(pt1[0]), int(pt1[1]))
                pt2 = (int(pt2[0]), int(pt2[1]))

                cv2.line(filtered_img, pt1, pt2, (0, 0, 255), 2)
        elif cls == 2:
            lines_in_box = []
            for i in range(rawLineNum):
                pt1 = (out[i, 0, 1] * scale_x, out[i, 0, 0] * scale_y)
                pt2 = (out[i, 1, 1] * scale_x, out[i, 1, 0] * scale_y)

                mid_pt = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

                if (x_min <= mid_pt[0] <= x_max) and (y_min <= mid_pt[1] <= y_max):
                    delta_x = pt2[0] - pt1[0]
                    delta_y = pt2[1] - pt1[1]
                    if delta_y == 0:
                        delta_y = 1e-6
                    if abs(delta_x / delta_y) < np.tan(np.deg2rad(30)):
                        lines_in_box.append((pt1, pt2))

            longest_line = get_longest_line(lines_in_box)
            if longest_line:
                pt1, pt2 = longest_line

                pt1 = np.array(pt1)
                pt2 = np.array(pt2)
                pt1[0] = np.clip(pt1[0], x_min, x_max)
                pt1[1] = np.clip(pt1[1], y_min + 15, y_max)
                pt2[0] = np.clip(pt2[0], x_min, x_max)
                pt2[1] = np.clip(pt2[1], y_min + 15, y_max)

                bbox_height = y_max - y_min - 30
                required_length = 0.98 * bbox_height
                line_length = abs(pt2[1] - pt1[1])

                if line_length < required_length:
                    x = (pt1[0] + pt2[0]) / 2

                    y1 = y_min + 15
                    y2 = y_max - 15
                    pt1 = np.array([x, y1])
                    pt2 = np.array([x, y2])

                if center_x < img.shape[1] / 2:
                    longest_cls2_left_line = (pt1, pt2)
                    found_cls2_left_line = True
                else:
                    longest_cls2_right_line = (pt1, pt2)
                    found_cls2_right_line = True

                pt1 = (int(pt1[0]), int(pt1[1]))
                pt2 = (int(pt2[0]), int(pt2[1]))

                cv2.line(filtered_img, pt1, pt2, (0, 0, 255), 2)

    if found_cls1_line and found_cls2_left_line:
        x1, y1 = longest_cls1_line[0]
        x2, y2 = longest_cls1_line[1]
        x3, y3 = longest_cls2_left_line[0]
        x4, y4 = longest_cls2_left_line[1]

        if y3 < y4:
            coordinates["longi_left_upper_x"] = x3
            coordinates["longi_left_upper_y"] = y3
        else:
            coordinates["longi_left_upper_x"] = x4
            coordinates["longi_left_upper_y"] = y4

        intersection = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
        if intersection:
            coordinates["longi_left_lower_x"] = intersection[0]
            coordinates["longi_left_lower_y"] = intersection[1]
        else:
            coordinates["longi_left_lower_x"] = 0
            coordinates["longi_left_lower_y"] = 0
    else:
        coordinates["longi_left_upper_x"] = 0
        coordinates["longi_left_upper_y"] = 0
        coordinates["longi_left_lower_x"] = 0
        coordinates["longi_left_lower_y"] = 0

    if found_cls1_line and found_cls2_right_line:
        x1, y1 = longest_cls1_line[0]
        x2, y2 = longest_cls1_line[1]
        x3, y3 = longest_cls2_right_line[0]
        x4, y4 = longest_cls2_right_line[1]

        if y3 < y4:
            coordinates["longi_right_upper_x"] = x3
            coordinates["longi_right_upper_y"] = y3
        else:
            coordinates["longi_right_upper_x"] = x4
            coordinates["longi_right_upper_y"] = y4

        intersection = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
        if intersection:
            coordinates["longi_right_lower_x"] = intersection[0]
            coordinates["longi_right_lower_y"] = intersection[1]
        else:
            coordinates["longi_right_lower_x"] = 0
            coordinates["longi_right_lower_y"] = 0
    else:
        coordinates["longi_right_upper_x"] = 0
        coordinates["longi_right_upper_y"] = 0
        coordinates["longi_right_lower_x"] = 0
        coordinates["longi_right_lower_y"] = 0

    json_data = {
        "is_collar": is_collar,
        "collar_type": collar_type,
        "is_hole_left": is_hole_left,
        "is_hole_right": is_hole_right,
        "hole_left_bounding_width": hole_left_bounding_width,
        "hole_right_bounding_width": hole_right_bounding_width,
        "coordinates": coordinates
    }

    json_data_converted = convert_numpy_types(json_data)

    airline_save_path = os.path.join(save_folder, f"{img_name}_airline.png")
    yolo_save_path = os.path.join(save_folder, f"{img_name}_yolo.png")
    filtered_save_path = os.path.join(save_folder, f"{img_name}_filtered.png")
    json_save_path = os.path.join(save_folder, f"{img_name}.json")

    json_save_path_for_coordinates_inference = os.path.join(save_folder_for_all_progoram, f"{img_name}_coordinates_inference.json")

    cv2.imwrite(airline_save_path, airline_only_img)
    cv2.imwrite(yolo_save_path, yolo_only_img)
    cv2.imwrite(filtered_save_path, filtered_img)

    with open(json_save_path, 'w') as f:
        json.dump(json_data_converted, f, indent=4)

    with open(json_save_path_for_coordinates_inference, 'w') as f:
        json.dump(json_data_converted, f, indent=4)
    img_count += 1
    print(f"이미지 및 JSON 저장됨:\n - AirLine 결과: {airline_save_path}\n - YOLO 결과: {yolo_save_path}\n - 필터링 결과: {filtered_save_path}\n - JSON 파일: {json_save_path}\n")
    print(f"이미지 카운트: {img_count}\n")
    
    # Ground Truth 검증 (오차 평균 부분)
    # Load ground truth and compute errors
    # ground_truth_filename = f"{img_name}_coordinates_answer.json"
    # ground_truth_filepath = os.path.join(ground_truth_folder, ground_truth_filename)

    # # Check if ground truth file exists
    # if os.path.exists(ground_truth_filepath):
    #     with open(ground_truth_filepath, 'r') as f:
    #         ground_truth_data = json.load(f)
    # else:
    #     print(f"Ground truth file not found for {img_name}, skipping error computation.")
    #     continue

    # Increment total images
    # total_images += 1

    # # Compare and collect errors
    # # Binary and categorical fields
    # fields_to_compare = ['is_collar', 'collar_type', 'is_hole_left', 'is_hole_right']

    # for field in fields_to_compare:
    #     total_field_counts[field] += 1
    #     pred_value = json_data_converted.get(field, None)
    #     gt_value = ground_truth_data.get(field, None)
    #     if pred_value is not None and gt_value is not None:
    #         if pred_value == gt_value:
    #             correct_counts[field] += 1
    #     else:
    #         print(f"Field {field} missing in prediction or ground truth for {img_name}")

    # # Numeric fields
    # numeric_fields = ['hole_left_bounding_width', 'hole_right_bounding_width']

    # for field in numeric_fields:
    #     total_field_counts[field] += 1
    #     pred_value = json_data_converted.get(field, None)
    #     gt_value = ground_truth_data.get(field, None)
    #     if pred_value is not None and gt_value is not None:
    #         error = abs(pred_value - gt_value)
    #         total_errors[field] += error
    #     else:
    #         print(f"Field {field} missing in prediction or ground truth for {img_name}")

    # # Coordinates
    # for field in coordinate_fields:
    #     total_field_counts['coordinates'][field] += 1
    #     pred_value = json_data_converted['coordinates'].get(field, None)
    #     gt_value = ground_truth_data['coordinates'].get(field, None)
    #     if pred_value is not None and gt_value is not None:
    #         error = abs(pred_value - gt_value)
    #         total_errors['coordinates'][field] += error
    #     else:
    #         print(f"Coordinate field {field} missing in prediction or ground truth for {img_name}")

print("프로세싱이 완료되었습니다.\n")

