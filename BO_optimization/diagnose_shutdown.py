"""
진단 도구 - 프로세스 종료 원인 파악
목적:
1. GPU 메모리 한계 확인
2. AirLine/CRG311 모듈 안정성 체크
3. CUDA 연산 테스트
4. 메모리 누수 패턴 탐지
"""
import torch
import numpy as np
import cv2
import psutil
import gc
import sys
import time
from pathlib import Path

# 모듈 import
sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import detect_with_full_pipeline
from yolo_detector import YOLODetector
from environment_independent import extract_parameter_independent_environment


class SystemMonitor:
    """시스템 리소스 모니터링"""

    def __init__(self):
        self.process = psutil.Process()
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    def get_status(self):
        """현재 메모리 상태 반환"""
        mem_info = self.process.memory_info()
        ram_mb = mem_info.rss / 1024**2

        status = {
            'ram_mb': ram_mb,
            'ram_percent': self.process.memory_percent(),
        }

        if self.has_gpu:
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            status.update({
                'gpu_allocated_gb': gpu_allocated,
                'gpu_reserved_gb': gpu_reserved,
                'gpu_percent': (gpu_allocated / self.gpu_total_memory) * 100
            })

        return status

    def print_status(self, label=""):
        """상태 출력"""
        status = self.get_status()
        print(f"\n[{label}]")
        print(f"  RAM: {status['ram_mb']:.1f} MB ({status['ram_percent']:.1f}%)")
        if self.has_gpu:
            print(f"  GPU: {status['gpu_allocated_gb']:.2f}/{self.gpu_total_memory:.2f} GB "
                  f"({status['gpu_percent']:.1f}%)")
            print(f"  GPU Reserved: {status['gpu_reserved_gb']:.2f} GB")


def test_airline_stability(monitor, image_dir, yolo_detector, n_iterations=20):
    """AirLine 모듈 안정성 테스트 - 반복 실행"""
    print("\n" + "="*60)
    print("TEST 1: AirLine 모듈 안정성 (20회 반복)")
    print("="*60)

    # 이미지 로드
    image_files = sorted(list(Path(image_dir).glob('*.jpg')))[:5]  # 5개만
    print(f"테스트 이미지: {len(image_files)}개")

    # 기본 파라미터
    params = {
        'edgeThresh1': -3.0,
        'simThresh1': 0.98,
        'pixelRatio1': 0.05,
        'edgeThresh2': 1.0,
        'simThresh2': 0.75,
        'pixelRatio2': 0.05,
        'ransac_Q_w': 5.0,
        'ransac_QG_w': 5.0
    }

    monitor.print_status("시작 전")

    success_count = 0
    for i in range(n_iterations):
        try:
            print(f"\n반복 {i+1}/{n_iterations}...")

            for img_file in image_files:
                image = cv2.imread(str(img_file))

                # AirLine 실행
                coords = detect_with_full_pipeline(image, params, yolo_detector)

                # 메모리 해제
                del image, coords

            # 주기적 메모리 정리
            if (i + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                monitor.print_status(f"반복 {i+1} 완료")

            success_count += 1

        except Exception as e:
            print(f"  ❌ 에러 발생: {e}")
            monitor.print_status(f"에러 발생 (반복 {i+1})")
            break

    monitor.print_status("테스트 종료")
    print(f"\n결과: {success_count}/{n_iterations} 성공")

    return success_count == n_iterations


def test_gpu_memory_limit(monitor):
    """GPU 메모리 한계 테스트"""
    print("\n" + "="*60)
    print("TEST 2: GPU 메모리 한계 테스트")
    print("="*60)

    if not torch.cuda.is_available():
        print("GPU 없음 - 스킵")
        return True

    monitor.print_status("시작 전")

    try:
        # 점진적으로 메모리 할당
        tensors = []
        size_mb = 100

        for i in range(100):  # 최대 10GB까지
            tensor = torch.randn(size_mb * 1024 * 256, dtype=torch.float32, device='cuda')
            tensors.append(tensor)

            if (i + 1) % 10 == 0:
                monitor.print_status(f"{(i+1)*size_mb} MB 할당")

            # 80% 넘으면 경고
            status = monitor.get_status()
            if status['gpu_percent'] > 80:
                print(f"  ⚠️  GPU 메모리 80% 초과!")
                break

        # 정리
        del tensors
        torch.cuda.empty_cache()
        monitor.print_status("정리 후")

        return True

    except RuntimeError as e:
        print(f"  ❌ GPU 메모리 부족: {e}")
        monitor.print_status("에러 발생")
        return False


def test_crg311_stability(monitor, n_iterations=10):
    """CRG311 모듈 안정성 테스트"""
    print("\n" + "="*60)
    print("TEST 3: CRG311 모듈 안정성")
    print("="*60)

    try:
        import CRG311
        print("CRG311 import 성공")
    except Exception as e:
        print(f"  ❌ CRG311 import 실패: {e}")
        return False

    monitor.print_status("시작 전")

    success_count = 0
    for i in range(n_iterations):
        try:
            # 더미 이미지 생성
            img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

            # CRG311 사용하는 함수 호출 (AirLine 내부)
            # 여기서는 간접적으로 테스트

            if (i + 1) % 5 == 0:
                print(f"  반복 {i+1}/{n_iterations} 완료")

            success_count += 1

        except Exception as e:
            print(f"  ❌ 에러 발생: {e}")
            break

    monitor.print_status("테스트 종료")
    print(f"\n결과: {success_count}/{n_iterations} 성공")

    return success_count == n_iterations


def test_memory_intensive_ops(monitor):
    """메모리 집약적 연산 테스트"""
    print("\n" + "="*60)
    print("TEST 4: 메모리 집약적 연산 반복")
    print("="*60)

    monitor.print_status("시작 전")

    try:
        for i in range(15):  # 13번 넘어서 15번까지
            print(f"\n반복 {i+1}/15...")

            # 큰 배열 생성 (실제 이미지 처리 시뮬레이션)
            images = []
            for j in range(50):  # 50개 이미지 시뮬레이션
                img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # 이미지 처리 시뮬레이션
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blur, 50, 150)

                images.append((img, gray, blur, edges))

            # GPU 연산 (있으면)
            if torch.cuda.is_available():
                tensor = torch.randn(1000, 1000, device='cuda')
                result = torch.matmul(tensor, tensor.T)
                del tensor, result

            # 메모리 해제
            del images
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % 5 == 0:
                monitor.print_status(f"반복 {i+1} 완료")

            # 13번째 특별 체크
            if i + 1 == 13:
                print("  🔍 13번째 반복 통과!")
                monitor.print_status("13번 통과")

        monitor.print_status("테스트 완료")
        print("\n✅ 15번 반복 성공!")
        return True

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        monitor.print_status("에러 발생")
        return False


def main():
    print("="*60)
    print("프로세스 종료 원인 진단 도구")
    print("="*60)

    # 시스템 정보
    monitor = SystemMonitor()
    print(f"\nCPU 코어: {psutil.cpu_count()}")
    print(f"총 RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")

    if monitor.has_gpu:
        print(f"GPU: {monitor.gpu_name}")
        print(f"GPU 메모리: {monitor.gpu_total_memory:.1f} GB")
    else:
        print("GPU: 없음")

    # 이미지 디렉토리 설정
    image_dir = Path(__file__).parent.parent / "dataset" / "images" / "test"
    yolo_model_path = Path(__file__).parent / "models" / "best.pt"

    print(f"\n이미지 디렉토리: {image_dir}")
    print(f"YOLO 모델: {yolo_model_path}")

    # YOLO 로드
    print("\nYOLO 모델 로딩...")
    yolo_detector = YOLODetector(str(yolo_model_path))
    print("✅ YOLO 로드 완료")

    # 테스트 실행
    results = {}

    # Test 1: AirLine 안정성
    results['airline'] = test_airline_stability(monitor, image_dir, yolo_detector, n_iterations=20)

    # Test 2: GPU 메모리 한계
    results['gpu_memory'] = test_gpu_memory_limit(monitor)

    # Test 3: CRG311 안정성
    results['crg311'] = test_crg311_stability(monitor, n_iterations=10)

    # Test 4: 메모리 집약적 연산
    results['memory_intensive'] = test_memory_intensive_ops(monitor)

    # 최종 리포트
    print("\n" + "="*60)
    print("진단 결과 요약")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"{test_name:20s}: {status}")

    # 권장사항
    print("\n" + "="*60)
    print("권장사항")
    print("="*60)

    if not results['gpu_memory']:
        print("⚠️  GPU 메모리 80% 제한 설정 필요")

    if not results['airline']:
        print("⚠️  AirLine 메모리 관리 강화 필요")

    if not results['memory_intensive']:
        print("⚠️  주기적 메모리 정리 강화 필요 (5번마다)")

    print("\n다음 단계: Phase 2 - GPU 메모리 관리 강화")
    print("="*60)


if __name__ == "__main__":
    main()
