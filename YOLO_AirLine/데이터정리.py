import os
import shutil

# 경로 설정
final_json_dir = r"C:\Users\user\Desktop\study\task\weld2025\weld2025_samsung_git_temp\testing\samsung2024\6_30_final_test_20250630_093301\strategy=QualityFocused_GenerousSim_px=300\final_json_results"
input_data_dir = r"C:\Users\user\Desktop\study\task\weld2025\weld2025_samsung_git_temp\testing\samsung2024\new_pendant_translator\inputData"
output_dir = r"C:\Users\user\Desktop\study\task\weld2025\weld2025_samsung_git_temp\testing\samsung2024\pendantTrainData"

# 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# inputData 폴더에 있는 파일 이름 목록 (확장자 포함)
input_filenames = set(os.listdir(input_data_dir))

# final_json_results 디렉터리에서 파일 이름 비교 후 복사
for filename in os.listdir(final_json_dir):
    if filename in input_filenames:
        src = os.path.join(final_json_dir, filename)
        dst = os.path.join(output_dir, filename)
        shutil.copy2(src, dst)
        print(f"Copied: {filename}")

print("작업 완료료")
