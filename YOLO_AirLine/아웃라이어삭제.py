import os
import json

# 폴더 경로 설정
output_dir = r"C:\Users\user\Desktop\study\task\weld2025\weld2025_samsung_git_temp\testing\samsung2024\pendantTrainData"

# 반복적으로 json 파일 확인
for filename in os.listdir(output_dir):
    if filename.lower().endswith('.json'):
        file_path = os.path.join(output_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # pixel_scalar에 0이 하나라도 있으면 삭제
            pixel_scalar = data.get("pixel_scalar", {})
            if any(value == 0 for value in pixel_scalar.values()):
                os.remove(file_path)
                print(f"Deleted (contains 0 in pixel_scalar): {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("검사 및 삭제 완료.")
