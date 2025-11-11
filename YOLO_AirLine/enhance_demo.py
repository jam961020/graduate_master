import argparse
import sys
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def laplacian_variance(gray_image: np.ndarray) -> float:
    """그레이스케일 이미지의 라플라시안 분산을 계산하여 블러 정도를 측정합니다."""
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def sharp_S(gray_blur: np.ndarray) -> float:
    """원본 코드의 S 메트릭 계산식 그대로 적용."""
    mu = float(gray_blur.mean())
    var_l = float(cv2.Laplacian(gray_blur, cv2.CV_64F).var())
    return var_l / (mu ** 2 + 1e-6) * 255.0 / (mu + 12.8)


def enhance(gray_blur: np.ndarray, S: float, t1: float = 2e-3, t2: float = 6e-3, clip_hi: float = 8.0) -> np.ndarray:
    """
    S에 따라 CLAHE clipLimit를 동적으로 선택. 충분히 샤프하면 가우시안 블러 반환.
    gray_blur은 이미 (3x3, 0.8) 가우시안 블러가 적용된 그레이 영상을 기대합니다.
    """
    if S < t1:
        clip = clip_hi
    elif S < t2:
        clip = 1.0 + (clip_hi - 1.0) * (t2 - S) / (t2 - t1)
    else:
        return cv2.GaussianBlur(gray_blur, (3, 3), 0.8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray_blur)


def enhance_color(bgr_image: np.ndarray, t1: float = 2e-3, t2: float = 6e-3, clip_hi: float = 8.0) -> Tuple[np.ndarray, float, float]:
    """
    L*a*b* 색상 공간으로 변환하여 L 채널만 동적 보정.
    반환값: (보정된 BGR, 원본 그레이 라플라시안 분산, S값)
    """
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # 그레이/샤프 측정은 L 채널 기반으로 수행
    l_blur = cv2.GaussianBlur(l_channel, (3, 3), 0.8)
    var = laplacian_variance(l_blur)
    S = sharp_S(l_blur)

    enhanced_l = enhance(l_blur, S, t1=t1, t2=t2, clip_hi=clip_hi)
    updated_lab_image = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(updated_lab_image, cv2.COLOR_LAB2BGR)
    return enhanced_bgr, var, S


def put_info_panel(image: np.ndarray, var: float, S: float, title: str) -> np.ndarray:
    """시연용: 좌상단에 라플라시안 분산과 S 지표, 타이틀을 크게 표기."""
    vis = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(vis, (10, 10), (10 + 540, 10 + 120), (0, 0, 0), -1)
    cv2.putText(vis, title, (24, 54), font, 1.2, (255, 255, 255), 3)
    cv2.putText(vis, f"Laplacian Var: {var:.2f}", (24, 94), font, 1.0, (0, 255, 255), 2)
    cv2.putText(vis, f"S metric: {S:.6f}", (280, 94), font, 1.0, (0, 255, 255), 2)
    return vis


def process_image(path: Path, out_dir: Path, show: bool = False, save_side_by_side: bool = True,
                  t1: float = 2e-3, t2: float = 6e-3, clip_hi: float = 8.0) -> None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[WARN] Failed to read image: {path}")
        return

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0.8)
    var_gray = laplacian_variance(gray_blur)
    S_gray = sharp_S(gray_blur)
    enhanced_gray = enhance(gray_blur, S_gray, t1=t1, t2=t2, clip_hi=clip_hi)

    enhanced_color, var_color, S_color = enhance_color(bgr, t1=t1, t2=t2, clip_hi=clip_hi)

    # 저장 경로 준비
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    # 저장: 그레이/컬러
    gray_out = out_dir / f"{stem}_gray_enh.png"
    color_out = out_dir / f"{stem}_color_enh.png"
    cv2.imwrite(str(gray_out), enhanced_gray)
    cv2.imwrite(str(color_out), enhanced_color)

    # 정보 오버레이 버전도 저장
    gray_vis = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    gray_vis = put_info_panel(gray_vis, var_gray, S_gray, "Grayscale Enhanced")
    color_vis = put_info_panel(enhanced_color, var_color, S_color, "LAB L-Channel Enhanced")

    vis_out = out_dir / f"{stem}_enh_vis.png"
    cv2.imwrite(str(vis_out), np.hstack([cv2.resize(bgr, (gray_vis.shape[1], gray_vis.shape[0])), gray_vis, color_vis]))

    print(f"[DONE] {path.name} ->\n  - Gray var={var_gray:.2f}, S={S_gray:.6f} -> {gray_out.name}\n  - Color var(L)={var_color:.2f}, S(L)={S_color:.6f} -> {color_out.name}")

    if show:
        cv2.imshow("Original | Gray Enhanced | Color Enhanced", np.hstack([cv2.resize(bgr, (gray_vis.shape[1], gray_vis.shape[0])), gray_vis, color_vis]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in EXTS])
    if input_path.is_file() and input_path.suffix.lower() in EXTS:
        return [input_path]
    return []


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Laplacian-variance based enhancement demo (GaussianBlur/CLAHE)")
    parser.add_argument("input", type=str, help="입력 이미지 파일 또는 디렉터리")
    parser.add_argument("--out", type=str, default="outputs/enhance_demo", help="결과 저장 디렉터리")
    parser.add_argument("--show", action="store_true", help="시각화 창 표시")
    parser.add_argument("--t1", type=float, default=2e-3, help="S 임계1 (CLAHE 최대 클립)")
    parser.add_argument("--t2", type=float, default=6e-3, help="S 임계2 (이상 시 가우시안 블러 선택)")
    parser.add_argument("--clip_hi", type=float, default=8.0, help="CLAHE 최대 clipLimit")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.out)

    images = collect_images(input_path)
    if not images:
        print(f"[ERROR] No images found in: {input_path}")
        return 1

    for img_path in images:
        try:
            process_image(img_path, out_dir, show=args.show, t1=args.t1, t2=args.t2, clip_hi=args.clip_hi)
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:])) 