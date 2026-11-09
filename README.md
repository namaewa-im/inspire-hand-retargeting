# Inspire Hand Retargeting

An experimental pipeline that uses an Intel RealSense D405 and MediaPipe to estimate human hand joint angles, visualize them in a real-time UI, and transmit them over UDP.

By default, this repository assumes installation at the following location in your home directory:

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

The six estimated joint values are ordered as follows:

1. Thumb CMC yaw
2. Thumb CMC pitch
3. Index MCP
4. Middle MCP
5. Ring MCP
6. Pinky MCP

## Repository Structure

When cloned into `~/maniskill`, the repository has the following structure:

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

Each file serves the following purpose:

- `hand_joint_estimator.py`  
  Detects hand landmarks in D405 images using MediaPipe and computes the six joint values.

- `d405_hand_control_ui.py`  
  Visualizes joint values and camera images in real time, provides controls for adjusting filter parameters, and transmits joint values over UDP.

- `filter.py`  
  Provides post-processing filters to reduce jitter in the estimated hand joint values.

- `udp_subscriber.py`  
  A debugging script that receives joint values over UDP and prints them to the terminal.

- `test/`  
  Contains test and experimental scripts for checking camera connectivity, MediaPipe inference, and the joint estimation pipeline.

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

Clone the repository into a directory named `maniskill` in your home directory:

```bash
cd ~
git clone https://github.com/namaewa-im/inspire-hand-retargeting.git maniskill
cd ~/maniskill
```

> The clone command will fail if `~/maniskill` already exists and is not empty. If that directory is already in use, rename it or choose a different destination before cloning.

Create a virtual environment and install the required packages:

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

The MediaPipe Hand Landmarker model file must be located at:

```text
~/maniskill/models/hand_landmarker.task
```

Create the model directory if it does not exist:

```bash
mkdir -p ~/maniskill/models
```

Then place the `hand_landmarker.task` file at the following path:

```text
~/maniskill/models/hand_landmarker.task
```

## Quick Start

First, navigate to the project root and activate the virtual environment:

```bash
cd ~/maniskill
source .venv/bin/activate
```

### Run the UI

```bash
python d405_hand_control_ui.py
```

### Check UDP Output

Open a separate terminal and run:

```bash
cd ~/maniskill
source .venv/bin/activate
python udp_subscriber.py
```

### Run the Estimator Directly

If `hand_joint_estimator.py` includes a standalone entry point, you can run it directly:

```bash
cd ~/maniskill
source .venv/bin/activate
python hand_joint_estimator.py
```

## Notes

- All commands assume that `~/maniskill` is the project root.
- Camera initialization may fail if the Intel RealSense D405 is not connected.
- The estimator will not run if the MediaPipe model file is missing from `~/maniskill/models/hand_landmarker.task`.
- Device permissions or the RealSense driver setup may prevent `pyrealsense2` from detecting the camera.
- If the UDP sender and receiver run on different computers, check the IP address and port settings.

## Status

This repository contains experimental code for vision-based human-to-robot hand retargeting. APIs, joint definitions, filtering parameters, and directory structure may change during development.
