import socket
import threading
import time
import csv
import keyboard
from collections import deque
from src.screen_capture import capture_screen
from ultralytics import YOLO
from src.config import MODEL_PATH, CONFIDENCE

# Configura o modelo YOLO
model = YOLO(MODEL_PATH)

# Socket UDP para receber dados do script de memória
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5005))

# Estado partilhado
last_action_time = time.time()
last_action = "hold"
latest_memory_data = None
enemy_shown = 0
health_history = deque(maxlen=1000)
log_file = "combined_log.csv"
lock = threading.Lock()

def detect_enemies_loop():
    global enemy_shown
    while True:
        img, _, _, _, _ = capture_screen()
        results = model(img, conf=CONFIDENCE)
        classes = results[0].boxes.cls.cpu().numpy()

        with lock:
            enemy_shown = int(any(cls in [2, 3] for cls in classes))

        time.sleep(0.1)

def receive_memory_data_loop():
    global latest_memory_data
    while True:
        data, _ = sock.recvfrom(1024)
        with lock:
            latest_memory_data = eval(data.decode())
            # Guarda timestamp + health atual para histórico
            timestamp = latest_memory_data[0]
            health = latest_memory_data[1]
            health_history.append((timestamp, health))

def keyboard_listener_loop():
    while True:
        if keyboard.is_pressed("r"):
            register_action("reload")
        elif keyboard.is_pressed("1"):
            register_action("change_primary")
        elif keyboard.is_pressed("2"):
            register_action("change_secondary")
        time.sleep(0.05)

def idle_checker_loop():
    while True:
        if time.time() - last_action_time >= 4:
            register_action("hold")
        time.sleep(1)

def get_health_3_seconds_ago(current_ts):
    for ts, hp in reversed(health_history):
        if current_ts - ts >= 3:
            return hp
    return None

def register_action(action):
    global last_action_time, last_action

    with lock:
        if latest_memory_data is None:
            return

        timestamp = latest_memory_data[0]
        prev_health = get_health_3_seconds_ago(timestamp)
        if prev_health is None:
            prev_health = latest_memory_data[1]  # fallback se não houver dado de 3s atrás

        log_entry = latest_memory_data + [prev_health, enemy_shown, action]
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_entry)

        print(f"[LOG] {log_entry}")

        last_action = action
        last_action_time = time.time()

def main():
    try:
        with open(log_file, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "health", "primary_ammo", "primary_in_mag",
                "secondary_ammo", "secondary_in_mag", "prev_health_3s",
                "enemy_shown", "action_took"
            ])
    except FileExistsError:
        pass

    threading.Thread(target=detect_enemies_loop, daemon=True).start()
    threading.Thread(target=receive_memory_data_loop, daemon=True).start()
    threading.Thread(target=keyboard_listener_loop, daemon=True).start()
    threading.Thread(target=idle_checker_loop, daemon=True).start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
