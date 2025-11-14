"""
용접선 라벨링 GUI 툴 (개선 버전)
- 8개 핵심 좌표를 마우스 클릭으로 표시
- 이미지별 저장 및 불러오기 지원
- 모든 점 배치 후 클릭 시 가장 가까운 점 자동 이동
"""
import cv2
import json
import numpy as np
from pathlib import Path
import argparse


class WeldingLabeler:
    def __init__(self, image_dir, output_json):
        self.image_dir = Path(image_dir)
        self.output_json = Path(output_json)
        
        # 이미지 목록
        self.images = sorted(list(self.image_dir.glob("*.jpg")) + 
                           list(self.image_dir.glob("*.png")))
        
        if not self.images:
            print(f"No images found in {self.image_dir}")
            exit(1)
        
        # 기존 라벨 로드
        self.labels = self.load_labels()
        
        # 현재 상태
        self.current_idx = 0
        self.current_image = None
        self.display_image = None
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        
        # 라벨링할 포인트들
        self.point_names = [
            "longi_left_lower",
            "longi_right_lower",
            "longi_left_upper",
            "longi_right_upper",
            "collar_left_lower",
            "collar_left_upper",
        ]
        
        self.point_colors = {
            "longi_left_lower": (0, 0, 255),    # 빨강
            "longi_right_lower": (0, 0, 255),
            "longi_left_upper": (255, 100, 0),  # 파랑
            "longi_right_upper": (255, 100, 0),
            "collar_left_lower": (0, 255, 0),   # 초록
            "collar_left_upper": (0, 255, 0),
        }
        
        self.current_point_idx = 0
        self.current_points = {}
        
        # UI 설정
        self.window_name = "Welding Line Labeling Tool"
        self.help_text = [
            "=== Controls ===",
            "Left Click: Place point / Move nearest point (after all placed)",
            "Right Click: Remove last point",
            "SPACE: Next point",
            "ENTER: Save and next image",
            "BACKSPACE: Previous image",
            "S: Save current labels",
            "R: Reset current image",
            "Z/X: Zoom in/out",
            "Arrow keys: Pan",
            "Q/ESC: Quit",
            "",
            "=== Points to Label ===",
            "1. longi_left_lower (Red)",
            "2. longi_right_lower (Red)",
            "3. longi_left_upper (Blue)",
            "4. longi_right_upper (Blue)",
            "5. collar_left_lower (Green)",
            "6. collar_left_upper (Green)",
        ]
    
    def load_labels(self):
        """기존 라벨 파일 로드"""
        if self.output_json.exists():
            with open(self.output_json, 'r') as f:
                return json.load(f)
        return {}
    
    def save_labels(self):
        """라벨 파일 저장"""
        with open(self.output_json, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"Labels saved to {self.output_json}")
    
    def load_current_image(self):
        """현재 이미지 로드"""
        img_path = self.images[self.current_idx]
        self.current_image = cv2.imread(str(img_path))
        
        if self.current_image is None:
            print(f"Failed to load {img_path}")
            return False
        
        # 기존 라벨이 있으면 로드
        img_name = img_path.stem
        if img_name in self.labels:
            coords = self.labels[img_name]["coordinates"]
            self.current_points = {
                "longi_left_lower": (coords["longi_left_lower_x"], coords["longi_left_lower_y"]),
                "longi_right_lower": (coords["longi_right_lower_x"], coords["longi_right_lower_y"]),
                "longi_left_upper": (coords["longi_left_upper_x"], coords["longi_left_upper_y"]),
                "longi_right_upper": (coords["longi_right_upper_x"], coords["longi_right_upper_y"]),
                "collar_left_lower": (coords["collar_left_lower_x"], coords["collar_left_lower_y"]),
                "collar_left_upper": (coords["collar_left_upper_x"], coords["collar_left_upper_y"]),
            }
            # 0인 포인트 제거
            self.current_points = {k: v for k, v in self.current_points.items() if v != (0, 0)}
            self.current_point_idx = len(self.current_points)
        else:
            self.current_points = {}
            self.current_point_idx = 0
        
        # 줌/팬 리셋
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        
        return True
    
    def get_current_point_name(self):
        """현재 찍어야 할 포인트 이름"""
        if self.current_point_idx < len(self.point_names):
            return self.point_names[self.current_point_idx]
        return None
    
    def draw_display(self):
        """디스플레이 이미지 생성"""
        img = self.current_image.copy()
        h, w = img.shape[:2]
        
        # 이미 찍은 포인트들 그리기
        for name, (x, y) in self.current_points.items():
            color = self.point_colors.get(name, (255, 255, 255))
            cv2.circle(img, (int(x), int(y)), 8, color, -1)
            cv2.circle(img, (int(x), int(y)), 10, (255, 255, 255), 2)
            
            # 라벨 표시
            label = name.replace("_", " ").upper()
            cv2.putText(img, label, (int(x) + 15, int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 줌 적용
        if self.zoom_level != 1.0:
            new_w = int(w * self.zoom_level)
            new_h = int(h * self.zoom_level)
            img = cv2.resize(img, (new_w, new_h))
        
        # 팬 적용 (크롭)
        h, w = img.shape[:2]
        x_start = max(0, self.pan_offset[0])
        y_start = max(0, self.pan_offset[1])
        x_end = min(w, x_start + self.current_image.shape[1])
        y_end = min(h, y_start + self.current_image.shape[0])
        
        if x_end > x_start and y_end > y_start:
            img = img[y_start:y_end, x_start:x_end]
        
        # 상태 정보 오버레이
        self.draw_info_panel(img)
        
        self.display_image = img
    
    def draw_info_panel(self, img):
        """정보 패널 그리기"""
        h, w = img.shape[:2]
        
        # 반투명 배경
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 텍스트
        y_offset = 35
        
        # 현재 이미지 정보
        img_name = self.images[self.current_idx].name
        cv2.putText(img, f"Image: {img_name}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(img, f"Progress: {self.current_idx + 1}/{len(self.images)}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        # 현재 포인트
        current_name = self.get_current_point_name()
        if current_name:
            color = self.point_colors.get(current_name, (255, 255, 255))
            cv2.putText(img, f"Next: {current_name}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(img, "Click to move nearest point", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30
        
        # 포인트 체크리스트
        cv2.putText(img, "Points:", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
        
        for i, name in enumerate(self.point_names):
            status = "[X]" if name in self.current_points else "[ ]"
            color = (0, 255, 0) if name in self.current_points else (100, 100, 100)
            text = f"{status} {i+1}. {name}"
            cv2.putText(img, text, (30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 18
        
        # 줌 정보
        y_offset += 10
        cv2.putText(img, f"Zoom: {self.zoom_level:.1f}x", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 줌/팬 역변환
            actual_x = int((x + self.pan_offset[0]) / self.zoom_level)
            actual_y = int((y + self.pan_offset[1]) / self.zoom_level)
            
            current_name = self.get_current_point_name()
            
            if current_name:
                # 아직 찍을 포인트가 남음: 새 포인트 추가
                self.current_points[current_name] = (actual_x, actual_y)
                self.current_point_idx += 1
                print(f"✓ Placed {current_name} at ({actual_x}, {actual_y})")
            else:
                # 모든 포인트 배치 완료: 가장 가까운 점 이동
                if self.current_points:
                    closest_name = None
                    min_dist = float('inf')
                    
                    # 가장 가까운 포인트 찾기
                    for name, (px, py) in self.current_points.items():
                        dist = np.sqrt((px - actual_x)**2 + (py - actual_y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_name = name
                    
                    if closest_name:
                        old_pos = self.current_points[closest_name]
                        self.current_points[closest_name] = (actual_x, actual_y)
                        print(f"↔ Moved {closest_name} from {old_pos} to ({actual_x}, {actual_y}) (distance: {min_dist:.1f}px)")
            
            self.draw_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 우클릭: 마지막 포인트 제거
            if self.current_points:
                last_name = list(self.current_points.keys())[-1]
                del self.current_points[last_name]
                self.current_point_idx = len(self.current_points)
                print(f"✗ Removed {last_name}")
                self.draw_display()
    
    def save_current_labels(self):
        """현재 이미지 라벨 저장"""
        img_name = self.images[self.current_idx].stem
        
        coords = {
            "longi_left_lower_x": 0, "longi_left_lower_y": 0,
            "longi_right_lower_x": 0, "longi_right_lower_y": 0,
            "longi_left_upper_x": 0, "longi_left_upper_y": 0,
            "longi_right_upper_x": 0, "longi_right_upper_y": 0,
            "collar_left_lower_x": 0, "collar_left_lower_y": 0,
            "collar_left_upper_x": 0, "collar_left_upper_y": 0,
        }
        
        for name, (x, y) in self.current_points.items():
            coords[f"{name}_x"] = int(x)
            coords[f"{name}_y"] = int(y)
        
        self.labels[img_name] = {
            "image": str(self.images[self.current_idx]),
            "coordinates": coords
        }
        
        print(f"💾 Saved labels for {img_name}")
    
    def next_image(self):
        """다음 이미지로"""
        self.save_current_labels()
        self.save_labels()
        
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.load_current_image()
            self.draw_display()
        else:
            print("🏁 Last image reached!")
    
    def prev_image(self):
        """이전 이미지로"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()
            self.draw_display()
    
    def reset_current(self):
        """현재 이미지 리셋"""
        self.current_points = {}
        self.current_point_idx = 0
        self.draw_display()
        print("🔄 Reset current image")
    
    def zoom_in(self):
        """줌 인"""
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        self.draw_display()
    
    def zoom_out(self):
        """줌 아웃"""
        self.zoom_level = max(self.zoom_level / 1.2, 0.5)
        self.draw_display()
    
    def pan(self, dx, dy):
        """팬"""
        self.pan_offset[0] += dx
        self.pan_offset[1] += dy
        self.draw_display()
    
    def run(self):
        """메인 루프"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 첫 이미지 로드
        if not self.load_current_image():
            return
        
        self.draw_display()
        
        print("\n" + "="*60)
        print("Welding Line Labeling Tool (Improved)")
        print("="*60)
        for line in self.help_text:
            print(line)
        print("="*60 + "\n")
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("👋 Quitting...")
                self.save_labels()
                break
            
            elif key == ord(' '):  # SPACE
                # 다음 포인트로
                if self.current_point_idx < len(self.point_names):
                    self.current_point_idx += 1
                    self.draw_display()
            
            elif key == 13:  # ENTER
                # 다음 이미지
                self.next_image()
            
            elif key == 8:  # BACKSPACE
                # 이전 이미지
                self.prev_image()
            
            elif key == ord('s'):
                # 저장
                self.save_current_labels()
                self.save_labels()
            
            elif key == ord('r'):
                # 리셋
                self.reset_current()
            
            elif key == ord('z'):
                # 줌 인
                self.zoom_in()
            
            elif key == ord('x'):
                # 줌 아웃
                self.zoom_out()
            
            elif key == 82:  # 위 화살표
                self.pan(0, -50)
            
            elif key == 84:  # 아래 화살표
                self.pan(0, 50)
            
            elif key == 81:  # 왼쪽 화살표
                self.pan(-50, 0)
            
            elif key == 83:  # 오른쪽 화살표
                self.pan(50, 0)
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Welding Line Labeling Tool")
    parser.add_argument("--image_dir", required=True, help="Directory containing images")
    parser.add_argument("--output", default="labels.json", help="Output JSON file")
    args = parser.parse_args()
    
    labeler = WeldingLabeler(args.image_dir, args.output)
    labeler.run()


if __name__ == "__main__":
    main()