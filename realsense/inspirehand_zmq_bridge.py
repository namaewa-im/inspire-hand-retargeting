#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from dataclasses import dataclass

import zmq


HEADER_SIZE = 1280
REALSENSE_FEATURES = (
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "pinky_mcp",
)
INSPIRE_ORDER = (
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotate",
)


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def build_hand_message(left: list[float], right: list[float]) -> bytes:
    fields = [
        {"name": "left_hand_joints", "dtype": "f32", "shape": [len(left)]},
        {"name": "right_hand_joints", "dtype": "f32", "shape": [len(right)]},
    ]
    header = {
        "v": 1,
        "endian": "le",
        "count": 1,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")

    header_bytes = header_json.ljust(HEADER_SIZE, b"\x00")
    payload = b"".join(struct.pack("<f", clamp01(value)) for value in left)
    payload += b"".join(struct.pack("<f", clamp01(value)) for value in right)
    return b"planner" + header_bytes + payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge realsense UDP hand joints to InspireHand ZMQ planner messages."
    )
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--zmq-bind", default="*")
    parser.add_argument("--zmq-port", type=int, default=5556)
    parser.add_argument(
        "--hand",
        choices=["left", "right", "both", "auto"],
        default="left",
        help="Which InspireHand target to drive. auto uses the UDP handedness field.",
    )
    parser.add_argument(
        "--idle-open",
        action="store_true",
        help="Continuously publish fully open commands until the first valid UDP packet arrives.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="Publish rate in Hz while waiting for new UDP packets.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print every N published commands. 0 disables periodic prints.",
    )
    return parser.parse_args()


def extract_realsense_values(msg: dict) -> dict[str, float] | None:
    joint_names = msg.get("joint_names")
    joint_values = msg.get("joint_values")
    if not isinstance(joint_names, list) or not isinstance(joint_values, list):
        return None

    mapping = {
        str(name).lower(): clamp01(value)
        for name, value in zip(joint_names, joint_values)
    }
    for name in REALSENSE_FEATURES:
        if name not in mapping:
            return None
    return mapping


def realsense_to_inspire_command(values: dict[str, float]) -> list[float]:
    # realsense: 0=open, 1=closed
    # inspire planner: 0=closed, 1=open
    return [
        1.0 - clamp01(values["pinky_mcp"]),
        1.0 - clamp01(values["ring_mcp"]),
        1.0 - clamp01(values["middle_mcp"]),
        1.0 - clamp01(values["index_mcp"]),
        1.0 - clamp01(values["thumb_cmc_pitch"]),
        1.0 - clamp01(values["thumb_cmc_yaw"]),
    ]


def pick_target_hands(msg: dict, hand_mode: str) -> tuple[bool, bool]:
    if hand_mode == "left":
        return True, False
    if hand_mode == "right":
        return False, True
    if hand_mode == "both":
        return True, True

    handedness = str(msg.get("handedness", "")).strip().lower()
    if handedness == "left":
        return True, False
    if handedness == "right":
        return False, True
    return True, False


@dataclass
class PlannerState:
    left: list[float]
    right: list[float]


def print_command(prefix: str, state: PlannerState) -> None:
    left_text = " ".join(f"{value:.2f}" for value in state.left)
    right_text = " ".join(f"{value:.2f}" for value in state.right)
    print(f"{prefix} L[{left_text}] R[{right_text}]")


def main() -> int:
    args = parse_args()

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind((args.udp_host, args.udp_port))
    udp_socket.setblocking(False)

    context = zmq.Context.instance()
    zmq_socket = context.socket(zmq.PUB)
    endpoint = f"tcp://{args.zmq_bind}:{args.zmq_port}"
    zmq_socket.bind(endpoint)

    open_hand = [1.0] * 6
    state = PlannerState(left=open_hand.copy(), right=open_hand.copy())
    publish_dt = 1.0 / max(args.rate, 1.0)
    publish_count = 0
    last_publish_time = 0.0
    has_valid_packet = False

    print(f"[inspirehand_zmq_bridge] UDP listening on {args.udp_host}:{args.udp_port}")
    print(f"[inspirehand_zmq_bridge] ZMQ PUB bound on {endpoint}")
    print(f"[inspirehand_zmq_bridge] target hand mode: {args.hand}")
    print("[inspirehand_zmq_bridge] mapping:")
    print("  realsense input order : thumb_yaw thumb_pitch index middle ring pinky")
    print("  inspire output order  : pinky ring middle index thumb_bend thumb_rotate")
    print("  realsense 0=open/1=closed -> inspire 1=open/0=closed")
    print("[inspirehand_zmq_bridge] receiver example:")
    print("  sudo ./build_local/pico_inspire_hand_compliant --host 127.0.0.1 --port 5556")

    try:
        while True:
            received_packet = False

            try:
                data, addr = udp_socket.recvfrom(65535)
                msg = json.loads(data.decode("utf-8"))
                values = extract_realsense_values(msg)
                if values is None:
                    print("[inspirehand_zmq_bridge] skipped packet: missing required joint fields")
                else:
                    command = realsense_to_inspire_command(values)
                    send_left, send_right = pick_target_hands(msg, args.hand)

                    if send_left:
                        state.left = command.copy()
                    else:
                        state.left = open_hand.copy()

                    if send_right:
                        state.right = command.copy()
                    else:
                        state.right = open_hand.copy()

                    has_valid_packet = True
                    received_packet = True
                    print_command(f"\n[from {addr[0]}:{addr[1]}]", state)
            except BlockingIOError:
                pass
            except json.JSONDecodeError as exc:
                print(f"[inspirehand_zmq_bridge] invalid JSON packet: {exc}")
            except Exception as exc:
                print(f"[inspirehand_zmq_bridge] failed to process UDP packet: {type(exc).__name__}: {exc}")

            now = time.monotonic()
            should_publish = received_packet or (now - last_publish_time >= publish_dt)
            if not should_publish:
                time.sleep(0.001)
                continue

            if not has_valid_packet and not args.idle_open:
                last_publish_time = now
                time.sleep(0.001)
                continue

            zmq_socket.send(build_hand_message(state.left, state.right))
            publish_count += 1
            last_publish_time = now

            if args.print_every > 0 and publish_count % args.print_every == 0:
                print_command(f"[pub #{publish_count}]", state)

    except KeyboardInterrupt:
        print("\n[inspirehand_zmq_bridge] interrupted")
        return 0
    finally:
        zmq_socket.close(0)
        udp_socket.close()


if __name__ == "__main__":
    raise SystemExit(main())
