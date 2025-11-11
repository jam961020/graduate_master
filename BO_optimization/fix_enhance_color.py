"""
detection_wrapper.py의 enhance_color 호출 수정
"""

wrapper_file = "BO_optimization/detection_wrapper.py"

with open(wrapper_file, 'r', encoding='utf-8') as f:
    content = f.read()

# pre_gray 파라미터 제거
old_code = "enhance_color(image, pre_gray=gray_blur)"
new_code = "enhance_color(image)"

if old_code in content:
    content = content.replace(old_code, new_code)
    
    with open(wrapper_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ detection_wrapper.py 수정 완료!")
    print(f"  변경: {old_code}")
    print(f"  →    {new_code}")
else:
    print("⚠️ 해당 코드를 찾을 수 없습니다.")
    print(f"찾는 코드: {old_code}")