# Inspire Hand Retargeting

Intel RealSense D405와 MediaPipe를 이용해 사람 손의 관절 상태를 추정하고, 실시간 UI와 UDP 통신을 통해 확인하는 실험용 파이프라인입니다.

이 저장소는 기본적으로 사용자 홈 디렉터리의 다음 위치에 설치하는 것을 가정합니다.

```text
~/maniskill
```

## Pipeline

```text
Intel RealSense D405
        ↓
MediaPipe Hand Landmarker
        ↓
6-DoF hand joint estimation
        ↓
Jitter filtering
        ↓
Real-time UI / UDP transmission
```

추정하는 6개 관절 값의 순서는 다음과 같습니다.

1. Thumb CMC yaw
2. Thumb CMC pitch
3. Index MCP
4. Middle MCP
5. Ring MCP
6. Pinky MCP

## Repository Structure

저장소를 `~/maniskill`에 클론하면 기본 구조는 다음과 같습니다.

```text
~/maniskill/
├── hand_joint_estimator.py
├── d405_hand_control_ui.py
├── filter.py
├── udp_subscriber.py
├── models/
│   └── hand_landmarker.task
└── test/
```

각 파일의 역할은 다음과 같습니다.

- `hand_joint_estimator.py`  
  D405 영상에서 MediaPipe 손 랜드마크를 추정하고 6개 관절 값을 계산합니다.

- `d405_hand_control_ui.py`  
  관절 값과 카메라 영상을 실시간으로 시각화하고, 필터 파라미터 조절과 UDP 송신을 수행합니다.

- `filter.py`  
  손 관절 추정값의 jitter를 줄이기 위한 후처리 필터를 제공합니다.

- `udp_subscriber.py`  
  UDP로 전송된 관절 값을 수신하여 터미널에 출력하는 디버그 스크립트입니다.

- `test/`  
  카메라 연결, MediaPipe 추론 및 관절 추정 파이프라인 확인을 위한 테스트·실험 스크립트입니다.

## Requirements

### Hardware

- Intel RealSense D405

### Software

- Python 3
- `pyrealsense2`
- `mediapipe`
- `opencv-python`
- `numpy`
- `PySide6`
- `pyqtgraph`

## Installation

홈 디렉터리에서 저장소를 `maniskill`이라는 이름으로 클론합니다.

```bash
cd ~
git clone https://github.com/namaewa-im/inspire-hand-retargeting.git maniskill
cd ~/maniskill
```

> `~/maniskill` 디렉터리가 이미 존재하면 위 명령은 실패합니다. 기존 디렉터리를 사용 중이라면 삭제하거나 이름을 변경한 뒤 클론해야 합니다.

가상환경을 생성하고 필요한 패키지를 설치합니다.

```bash
cd ~/maniskill

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
    pyrealsense2 \
    mediapipe \
    opencv-python \
    numpy \
    PySide6 \
    pyqtgraph
```

## MediaPipe Model

MediaPipe Hand Landmarker 모델 파일이 다음 경로에 있어야 합니다.

```text
~/maniskill/models/hand_landmarker.task
```

모델 디렉터리가 없다면 생성합니다.

```bash
mkdir -p ~/maniskill/models
```

그다음 `hand_landmarker.task` 파일을 아래 위치에 배치합니다.

```bash
~/maniskill/models/hand_landmarker.task
```

## Quick Start

먼저 프로젝트 루트로 이동하고 가상환경을 활성화합니다.

```bash
cd ~/maniskill
source .venv/bin/activate
```

### Run the UI

```bash
python d405_hand_control_ui.py
```

### Check UDP Output

별도의 터미널을 열어 다음 명령을 실행합니다.

```bash
cd ~/maniskill
source .venv/bin/activate
python udp_subscriber.py
```

### Run the Estimator Directly

`hand_joint_estimator.py`에 직접 실행 코드가 포함되어 있는 경우 다음과 같이 실행할 수 있습니다.

```bash
cd ~/maniskill
source .venv/bin/activate
python hand_joint_estimator.py
```

## Notes

- 모든 명령은 기본적으로 `~/maniskill`을 프로젝트 루트로 가정합니다.
- Intel RealSense D405가 연결되어 있지 않으면 카메라 초기화 과정에서 오류가 발생할 수 있습니다.
- MediaPipe 모델 파일이 `~/maniskill/models/hand_landmarker.task`에 없으면 estimator가 실행되지 않습니다.
- 카메라 장치 권한이나 RealSense 드라이버 설치 상태에 따라 `pyrealsense2`가 카메라를 찾지 못할 수 있습니다.
- UDP 송신과 수신을 다른 컴퓨터에서 실행하는 경우 IP 주소와 포트 설정을 확인해야 합니다.

## Status

This repository contains experimental code for vision-based human-to-robot hand retargeting. APIs, joint definitions, filtering parameters, and directory structure may change during development.
