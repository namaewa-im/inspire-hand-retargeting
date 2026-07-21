import json
import socket
import os
  
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5005))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"listening on {HOST}:{PORT}")

while True:
    data, addr = sock.recvfrom(65535)
    msg = json.loads(data.decode("utf-8"))

    joint_values = msg["joint_values"]
    joint_names = msg["joint_names"]
    timestamp_s = msg["timestamp_s"]

    print("from:", addr)
    print("t:", timestamp_s)
    print(dict(zip(joint_names, joint_values)))

