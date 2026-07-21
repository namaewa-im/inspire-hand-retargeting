import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import mani_skill.envs


ROBOT_UID = "fixed_inspire_hand_left"
CONTROL_MODE = "pd_joint_pos"
KEYFRAME = "palm_side"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5005))
CAMERA_STATE_PATH = Path(__file__).with_name(".simulator_subscriber_camera.json")

FINGER_FEATURES = (
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "pinky_mcp",
)
ACTION_FINGER_START = 2


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    return array.reshape(-1)


def set_keyframe(base_env: Any, keyframe_name: str) -> None:
    keyframe = base_env.agent.keyframes[keyframe_name]

    if keyframe.qpos is not None:
        base_env.agent.robot.set_qpos(keyframe.qpos)

    if keyframe.qvel is not None:
        base_env.agent.robot.set_qvel(keyframe.qvel)

    base_env.agent.robot.set_pose(keyframe.pose)
    base_env.agent.controller.reset()

    if base_env.gpu_sim_enabled:
        base_env.scene._gpu_apply_all()
        base_env.scene.px.gpu_update_articulation_kinematics()
        base_env.scene._gpu_fetch_all()


def infer_finger_ranges(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    if action_low.size == 8:
        finger_low = action_low[ACTION_FINGER_START:]
        finger_high = action_high[ACTION_FINGER_START:]
    elif action_low.size == 6:
        finger_low = action_low
        finger_high = action_high
    else:
        raise RuntimeError(
            f"Unexpected action size {action_low.size}. Expected 6 or 8."
        )

    finger_open_q = np.maximum(finger_low, finger_high)
    finger_closed_q = np.minimum(finger_low, finger_high)
    return finger_open_q.astype(np.float32), finger_closed_q.astype(np.float32)


def map_normalized_to_qpos(
    normalized_values: np.ndarray,
    finger_open_q: np.ndarray,
    finger_closed_q: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(normalized_values, 0.0, 1.0).astype(np.float32)
    return finger_open_q + clipped * (finger_closed_q - finger_open_q)


def make_action(
    env: gym.Env,
    finger_qpos: np.ndarray,
) -> np.ndarray:
    action_shape = env.action_space.shape
    flat_dim = int(np.prod(action_shape))

    if flat_dim == 8:
        action = np.zeros(8, dtype=np.float32)
        action[ACTION_FINGER_START:] = finger_qpos
    elif flat_dim == 6:
        action = finger_qpos.astype(np.float32, copy=True)
    else:
        raise RuntimeError(
            f"Unexpected action size {flat_dim}. Expected 6 or 8."
        )

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    action = np.clip(action, low, high)
    return action.reshape(action_shape)


def extract_normalized_targets(msg: dict[str, Any]) -> np.ndarray | None:
    joint_names = msg.get("joint_names", [])
    joint_values = msg.get("joint_values", [])

    if not isinstance(joint_names, list) or not isinstance(joint_values, list):
        return None

    incoming = {
        str(name).lower(): float(value)
        for name, value in zip(joint_names, joint_values)
    }

    targets = []
    for feature in FINGER_FEATURES:
        if feature not in incoming:
            return None
        targets.append(incoming[feature])

    return np.asarray(targets, dtype=np.float32)


def print_mapping(finger_open_q: np.ndarray, finger_closed_q: np.ndarray) -> None:
    print("Finger mapping:")
    print("  normalized 0.0 = open")
    print("  normalized 1.0 = closed")
    print("  order: thumb_cmc_yaw, thumb_cmc_pitch, index_mcp, middle_mcp, ring_mcp, pinky_mcp")
    print("  idx  feature            q_open       q_closed")
    for index, feature in enumerate(FINGER_FEATURES):
        print(
            f"  {index:02d}   {feature:16s} "
            f"{finger_open_q[index]:+10.5f} "
            f"{finger_closed_q[index]:+10.5f}"
        )


def load_saved_camera_state() -> dict[str, list[float]] | None:
    if not CAMERA_STATE_PATH.exists():
        return None

    try:
        data = json.loads(CAMERA_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to read camera state: {type(exc).__name__}: {exc}")
        return None

    position = data.get("position")
    rotation = data.get("rotation")
    if not isinstance(position, list) or not isinstance(rotation, list):
        return None
    if len(position) != 3 or len(rotation) != 4:
        return None

    return {
        "position": [float(value) for value in position],
        "rotation": [float(value) for value in rotation],
    }


def save_camera_state(viewer: Any) -> None:
    window = getattr(viewer, "window", None)
    if window is None:
        return
    if not hasattr(window, "get_camera_position"):
        return
    if not hasattr(window, "get_camera_rotation"):
        return

    position = [float(value) for value in window.get_camera_position()]
    rotation = [float(value) for value in window.get_camera_rotation()]
    payload = {
        "position": position,
        "rotation": rotation,
    }

    try:
        CAMERA_STATE_PATH.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"Failed to save camera state: {type(exc).__name__}: {exc}")


def restore_camera_state(viewer: Any) -> None:
    state = load_saved_camera_state()
    if state is None:
        return

    window = getattr(viewer, "window", None)
    if window is None:
        return

    restored = False

    if hasattr(window, "set_camera_position"):
        window.set_camera_position(state["position"])
        restored = True

    if hasattr(window, "set_camera_rotation"):
        window.set_camera_rotation(state["rotation"])
        restored = True

    if restored:
        print(f"Restored camera view from {CAMERA_STATE_PATH.name}")
    else:
        print("Saved camera view found, but this viewer does not expose restore setters.")


def main() -> None:
    env = gym.make(
        "Empty-v1",
        robot_uids=ROBOT_UID,
        control_mode=CONTROL_MODE,
        obs_mode="none",
        reward_mode="none",
        render_mode="human",
        sim_config={
            "sim_freq": 100,
            "control_freq": 20,
        },
    )

    env.reset(seed=0)
    base_env = env.unwrapped
    set_keyframe(base_env, KEYFRAME)

    finger_open_q, finger_closed_q = infer_finger_ranges(env)
    print(f"Spawned robot: {ROBOT_UID}")
    print(f"Control mode: {CONTROL_MODE}")
    print(f"Listening on UDP {HOST}:{PORT}")
    print_mapping(finger_open_q, finger_closed_q)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.setblocking(False)

    viewer = env.render()
    viewer.paused = False
    restore_camera_state(viewer)
    last_action = make_action(env, finger_open_q)
    last_camera_save_time = 0.0

    while not viewer.closed:
        try:
            data, addr = sock.recvfrom(65535)
            msg = json.loads(data.decode("utf-8"))
            normalized_targets = extract_normalized_targets(msg)

            if normalized_targets is None:
                print("Skipped packet: missing one or more required joint names.")
                env.render()
                continue

            finger_qpos = map_normalized_to_qpos(
                normalized_targets,
                finger_open_q,
                finger_closed_q,
            )
            last_action = make_action(env, finger_qpos)
            env.step(last_action)

            print(f"\nfrom: {addr}")
            for index, feature in enumerate(FINGER_FEATURES):
                print(
                    f"  {feature:16s} norm={normalized_targets[index]:.3f} "
                    f"q={finger_qpos[index]:+.5f}"
                )

        except BlockingIOError:
            env.step(last_action)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON packet: {exc}")
        except Exception as exc:
            print(f"Failed to apply packet: {type(exc).__name__}: {exc}")

        env.render()
        now = time.monotonic()
        if now - last_camera_save_time >= 1.0:
            save_camera_state(viewer)
            last_camera_save_time = now

    save_camera_state(viewer)
    sock.close()
    env.close()


if __name__ == "__main__":
    main()
