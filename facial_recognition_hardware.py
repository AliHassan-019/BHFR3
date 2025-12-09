import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
import serial

# ================== CONFIG ==================

ENCODINGS_FILE = "encodings.pickle"

# UART config
UART_PORT = "/dev/ttyAMA0"   # keep what is working for you
UART_BAUD = 115200
DEBUG_UART = False           # set True if you want to see messages in terminal

# Camera config
FRAME_SIZE = (1280, 720)     # smaller than 1920x1080 -> faster
CV_SCALER = 4                # downscale factor for face_recognition
PROCESS_EVERY_N_FRAMES = 3   # run detection every N frames, reuse in between

# ================== LOAD ENCODINGS ==================

print("[INFO] Loading encodings...")
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]
print(f"[INFO] Loaded {len(known_face_names)} known faces.")

# ================== UART SETUP ==================

print(f"[INFO] Opening UART port {UART_PORT} @ {UART_BAUD} baud...")
try:
    uart = serial.Serial(
        UART_PORT,
        UART_BAUD,
        timeout=0.0  # non-blocking read
    )
    print("[INFO] UART opened successfully.")
except Exception as e:
    print("[ERROR] Could not open UART:", e)
    uart = None

# ================== CAMERA SETUP ==================

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": FRAME_SIZE}
    )
)
picam2.start()

# ================== STATE ==================

frame_count = 0
start_time = time.time()
fps = 0.0
frame_index = 0

# Cache last computed faces to reuse between heavy detections
last_face_infos = []  # same format as process_frame returns


# ================== FUNCTIONS ==================

def process_frame(frame):
    """
    Detect & recognize faces on a single frame.
    Returns a list of dicts:
      {
        "name": str,
        "bbox": (left, top, right, bottom),
        "center": (cx, cy),
        "area": int
      }
    All coordinates are in FULL-RESOLUTION pixels.
    """
    # Downscale frame for faster processing
    small_frame = cv2.resize(
        frame, (0, 0), fx=1.0 / CV_SCALER, fy=1.0 / CV_SCALER
    )
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations in downscaled frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    if not face_locations:
        return []

    # Compute encodings for each detected face
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations, model="small"
    )

    h_full, w_full, _ = frame.shape
    face_infos = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale back to full resolution
        top_f = int(top * CV_SCALER)
        right_f = int(right * CV_SCALER)
        bottom_f = int(bottom * CV_SCALER)
        left_f = int(left * CV_SCALER)

        # Clamp to frame boundaries
        top_f = max(0, min(top_f, h_full - 1))
        bottom_f = max(0, min(bottom_f, h_full - 1))
        left_f = max(0, min(left_f, w_full - 1))
        right_f = max(0, min(right_f, w_full - 1))

        width = right_f - left_f
        height = bottom_f - top_f
        area = max(0, width * height)

        cx = left_f + width // 2
        cy = top_f + height // 2

        face_infos.append(
            {
                "name": name,
                "bbox": (left_f, top_f, right_f, bottom_f),
                "center": (cx, cy),
                "area": area,
            }
        )

    return face_infos


def send_first_face_over_uart(face_infos):
    """
    Send data for the FIRST detected face over UART.

    Format:
      FACE,<name>,<cx>,<cy>,<width>,<height>\\n
    If no face:
      NOFACE\\n
    """
    if uart is None or not uart.is_open:
        return

    try:
        if face_infos:
            fc = face_infos[0]
            name = fc["name"]
            (cx, cy) = fc["center"]
            (left, top, right, bottom) = fc["bbox"]
            width = right - left
            height = bottom - top

            safe_name = str(name).replace(",", "_")

            message = f"FACE,{safe_name},{cx},{cy},{width},{height}\n"
        else:
            message = "NOFACE\n"

        uart.write(message.encode("utf-8"))
        if DEBUG_UART:
            print(message.strip())
    except Exception as e:
        if DEBUG_UART:
            print("[UART ERROR]", e)


def draw_results(frame, face_infos):
    """
    Draw rectangles and names for all faces on the frame.
    """
    if not face_infos:
        return frame

    max_area = max(fi["area"] for fi in face_infos)

    for fi in face_infos:
        name = fi["name"]
        (left, top, right, bottom) = fi["bbox"]
        area = fi["area"]

        if area == max_area:
            color = (0, 255, 0)    # green -> largest face
        else:
            color = (0, 165, 255)  # orange -> others

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(
            frame,
            (left, top - 30),
            (right, top),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            name,
            (left + 5, top - 8),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1,
        )

    return frame


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
        fps = elapsed and fps or 0
    return fps


# ================== MAIN LOOP ==================

try:
    while True:
        frame = picam2.capture_array()
        frame_index += 1

        # Only run heavy face_recognition every N frames
        if frame_index % PROCESS_EVERY_N_FRAMES == 0:
            last_face_infos = process_frame(frame)

        # Use last_face_infos for UART + drawing on every frame
        send_first_face_over_uart(last_face_infos)
        display_frame = draw_results(frame, last_face_infos)

        current_fps = calculate_fps()
        cv2.putText(
            display_frame,
            f"FPS: {current_fps:.1f}",
            (display_frame.shape[1] - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Video", display_frame)

        if cv2.waitKey(1) == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    if uart is not None and uart.is_open:
        uart.close()
