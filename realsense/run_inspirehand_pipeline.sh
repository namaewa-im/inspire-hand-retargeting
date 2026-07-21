#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DFX_ROOT="${REPO_ROOT}/dfx_inspire_service-master"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
UI_SCRIPT="${SCRIPT_DIR}/d405_hand_control_ui.py"
BRIDGE_SCRIPT="${SCRIPT_DIR}/inspirehand_zmq_bridge.py"
PICO_BIN="${DFX_ROOT}/build_local/pico_inspire_hand_compliant"

UDP_HOST="${UDP_HOST:-127.0.0.1}"
UDP_PORT="${UDP_PORT:-5005}"
ZMQ_HOST="${ZMQ_HOST:-127.0.0.1}"
ZMQ_PORT="${ZMQ_PORT:-5556}"
HAND_MODE="${HAND_MODE:-left}"
PREFERRED_HAND="${PREFERRED_HAND:-Left}"
FILTER_STRENGTH="${FILTER_STRENGTH:-0.286}"
JITTER="${JITTER:-true}"
USE_SUDO="${USE_SUDO:-1}"
PICO_EXTRA_ARGS=("${@}")

BRIDGE_PID=""
PICO_PID=""

cleanup() {
  local exit_code=$?

  if [[ -n "${BRIDGE_PID}" ]] && kill -0 "${BRIDGE_PID}" 2>/dev/null; then
    kill "${BRIDGE_PID}" 2>/dev/null || true
    wait "${BRIDGE_PID}" 2>/dev/null || true
  fi

  if [[ -n "${PICO_PID}" ]] && kill -0 "${PICO_PID}" 2>/dev/null; then
    kill "${PICO_PID}" 2>/dev/null || true
    wait "${PICO_PID}" 2>/dev/null || true
  fi

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

stop_stale_bridge() {
  local pids=()
  local pid=""

  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    [[ "${pid}" == "$$" ]] && continue
    pids+=("${pid}")
  done < <(
    pgrep -f \
      "${BRIDGE_SCRIPT} --udp-host ${UDP_HOST} --udp-port ${UDP_PORT} --zmq-bind ${ZMQ_HOST} --zmq-port ${ZMQ_PORT}"
  )

  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "[pipeline] stopping stale bridge on ${UDP_HOST}:${UDP_PORT}: ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true

  local waited=0
  while (( waited < 20 )); do
    local any_alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    (( any_alive == 0 )) && return 0
    sleep 0.1
    ((waited += 1))
  done

  echo "[pipeline] failed to stop stale bridge: ${pids[*]}" >&2
  return 1
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${UI_SCRIPT}" ]]; then
  echo "UI script not found: ${UI_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${BRIDGE_SCRIPT}" ]]; then
  echo "Bridge script not found: ${BRIDGE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -x "${PICO_BIN}" ]]; then
  echo "Pico service binary not found or not executable: ${PICO_BIN}" >&2
  exit 1
fi

echo "[pipeline] UDP  : ${UDP_HOST}:${UDP_PORT}"
echo "[pipeline] ZMQ  : ${ZMQ_HOST}:${ZMQ_PORT}"
echo "[pipeline] Hand : ${HAND_MODE}"
echo "[pipeline] UI preferred hand: ${PREFERRED_HAND}"
echo

stop_stale_bridge

echo "[pipeline] starting bridge"
"${PYTHON_BIN}" "${BRIDGE_SCRIPT}" \
  --udp-host "${UDP_HOST}" \
  --udp-port "${UDP_PORT}" \
  --zmq-bind "${ZMQ_HOST}" \
  --zmq-port "${ZMQ_PORT}" \
  --hand "${HAND_MODE}" \
  --idle-open &
BRIDGE_PID=$!

sleep 0.5

if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
  echo "[pipeline] bridge exited during startup; check port conflicts or bridge logs above" >&2
  wait "${BRIDGE_PID}" || true
  exit 1
fi

echo "[pipeline] starting pico service"
if [[ "${USE_SUDO}" == "1" ]]; then
  sudo "${PICO_BIN}" \
    --host "${ZMQ_HOST}" \
    --port "${ZMQ_PORT}" \
    "${PICO_EXTRA_ARGS[@]}" &
else
  "${PICO_BIN}" \
    --host "${ZMQ_HOST}" \
    --port "${ZMQ_PORT}" \
    "${PICO_EXTRA_ARGS[@]}" &
fi
PICO_PID=$!

sleep 0.5

echo "[pipeline] starting D405 UI"
"${PYTHON_BIN}" "${UI_SCRIPT}" \
  --preferred-hand "${PREFERRED_HAND}" \
  --filter-strength "${FILTER_STRENGTH}" \
  --jitter "${JITTER}" \
  --udp-host "${UDP_HOST}" \
  --udp-port "${UDP_PORT}"
