# realsense

Intel RealSense D405 카메라와 MediaPipe를 사용해 손 관절 값을 추정하고,
UI 및 UDP 전송으로 확인할 수 있는 실험용 스크립트 모음입니다.

## 구성

- `hand_joint_estimator.py`: D405 영상에서 손 랜드마크를 추정하고 6개 관절 값을 계산합니다.
- `d405_hand_control_ui.py`: 실시간 시각화, 필터 조절, UDP 송신 기능이 있는 UI입니다.
- `filter.py`: 손 관절 추정값 후처리 필터입니다.
- `udp_subscriber.py`: UDP로 송신된 관절 값을 수신해 출력하는 간단한 디버그 스크립트입니다.
- `test/`: 카메라 및 추정 파이프라인 확인용 테스트/실험 스크립트입니다.

## 요구 사항

- Intel RealSense D405
- Python 3
- `pyrealsense2`
- `mediapipe`
- `opencv-python`
- `numpy`
- `PySide6`
- `pyqtgraph`

또한 MediaPipe 모델 파일이 아래 경로에 있어야 합니다.

```text
~/maniskill/models/hand_landmarker.task
```

## 빠른 시작

UI 실행:

```bash
python d405_hand_control_ui.py
```

UDP 수신 확인:

```bash
python udp_subscriber.py
```

## 비고

- 이 디렉터리는 단일 패키지보다는 실험/개발용 스크립트 모음에 가깝습니다.
- 카메라 연결 및 모델 파일 경로가 맞지 않으면 실행 시 오류가 발생할 수 있습니다.
