#!/usr/bin/env python3
import cv2
import face_recognition
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from gpiozero import LED
import serial

# ===================== SETTINGS =====================
CV_SCALER = 4                   # downscale factor for recognition
UART_PORT = "/dev/serial0"      # UART0 on Pi GPIO14/15
UART_BAUD = 115200              # MUST match ESP32
UART_WRITE_TIMEOUT = 0.02       # non-blocking-ish writes
SEND_DEBUG_EVERY = 15           # print "sent" message every N frames

AUTHORIZED_NAMES = ["john", "alice", "bob"]  # example

# ===================== LOAD ENCODINGS =====================
print("[PI] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# ===================== CAMERA SETUP =====================
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (1920, 1080)}
    )
)
picam2.start()
print("[PI] Camera started at 1920x1080")

# ===================== GPIO =====================
output = LED(14)  # example pin for "authorized" indicator

# ===================== UART INIT =====================
uart = None
try:
    uart = serial.Serial(
        port=UART_PORT,
        baudrate=UART_BAUD,
        timeout=0,
        write_timeout=UART_WRITE_TIMEOUT,
    )
    print(f"[PI] UART opened on {UART_PORT} at {UART_BAUD} baud")
except Exception as e:
    print(f"[PI][ERROR] Could not open UART {UART_PORT}: {e}")
    print("[PI] Continuing without UART output...")

# ===================== GLOBALS =====================
frame_count = 0
start_time = time.time()
fps = 0.0
face_coords = []   # list of dicts for current frame


# -------------- UART helper --------------
def send_uart_line(line: str, is_debug=False):
    """Send a single text line over UART with newline."""
    global frame_count
    if uart is None or not uart.is_open:
        return
    try:
        uart.write(line.encode("utf-8"))
        if is_debug:
            print(f"[PI][UART] Sent: {line.strip()}")
    except Exception as e:
        print(f"[PI][UART ERROR] {e}")


def send_faces_over_uart(faces):
    """
    Send nearest face (largest area) if any,
    else send NOFACE.
    Format:
      FACE:<name>,cx,cy,left,top,right,bottom,area\n
      NOFACE\n
    """
    global frame_count

    if not faces:
        send_uart_line("NOFACE\n", is_debug=(frame_count % SEND_DEBUG_EVERY == 0))
        return

    # Choose "nearest" face = biggest bounding box area
    nearest = max(faces, key=lambda fc: fc["area"])
    name = nearest["name"]
    (left, top, right, bottom) = nearest["bbox"]
    (cx, cy) = nearest["center"]
    area = nearest["area"]

    line = f"FACE:{name},{cx},{cy},{left},{top},{right},{bottom},{area}\n"
    send_uart_line(line, is_debug=(frame_count % SEND_DEBUG_EVERY == 0))


# -------------- Face detection --------------
def process_frame(frame):
    """
    Detect & recognize faces.
    Update global face_coords and send over UART.
    """
    global face_coords

    # Downscale for speed
    small = cv2.resize(frame, (0, 0), fx=1 / CV_SCALER, fy=1 / CV_SCALER)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb_small)
    encodings = face_recognition.face_encodings(rgb_small, locations, model="small")

    names = []
    authorized_detected = False

    for enc in encodings:
        matches = face_recognition.compare_faces(known_face_encodings, enc)
        name = "Unknown"

        distances = face_recognition.face_distance(known_face_encodings, enc)
        if len(distances) > 0:
            best = np.argmin(distances)
            if matches[best]:
                name = known_face_names[best]
                if name in AUTHORIZED_NAMES:
                    authorized_detected = True

        names.append(name)

    # GPIO based on authorized face
    if authorized_detected:
        output.on()
    else:
        output.off()

    # Build full-res coordinates list
    face_coords = []
    h, w, _ = frame.shape
    for (top, right, bottom, left), name in zip(locations, names):
        top_full = max(0, min(top * CV_SCALER, h - 1))
        bottom_full = max(0, min(bottom * CV_SCALER, h - 1))
        left_full = max(0, min(left * CV_SCALER, w - 1))
        right_full = max(0, min(right * CV_SCALER, w - 1))

        width = right_full - left_full
        height = bottom_full - top_full
        area = max(0, width * height)
        cx = left_full + width // 2
        cy = top_full + height // 2

        face_coords.append(
            {
                "name": name,
                "bbox": (left_full, top_full, right_full, bottom_full),
                "center": (cx, cy),
                "area": area,
            }
        )

    # Send to ESP32
    send_faces_over_uart(face_coords)


def draw_results(frame):
    if not face_coords:
        return frame
    max_area = max(fc["area"] for fc in face_coords)
    for fc in face_coords:
        name = fc["name"]
        left, top, right, bottom = fc["bbox"]
        area = fc["area"]

        if area == max_area:
            color = (0, 255, 0)     # nearest = green
        else:
            color = (0, 165, 255)   # others = orange

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return frame


def update_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
        print(f"[PI] FPS: {fps:.1f}")
    return fps


# ===================== MAIN LOOP =====================
try:
    print("[PI] Starting main loop...")
    while True:
        frame = picam2.capture_array()
        process_frame(frame)
        frame = draw_results(frame)
        current_fps = update_fps()

        cv2.putText(
            frame,
            f"FPS: {current_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    print("[PI] Shutting down...")
    cv2.destroyAllWindows()
    picam2.stop()
    output.off()
    if uart is not None and uart.is_open:
        uart.close()
