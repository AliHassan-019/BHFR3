import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle

# ============ LOAD ENCODINGS ============
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# ============ CAMERA SETUP ============
picam2 = Picamera2()
# Lower resolution for better performance
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (1280, 720)}
    )
)
picam2.start()

# ============ GLOBALS & SETTINGS ============
cv_scaler = 4  # downscale factor for recognition
process_every_n_frames = 3  # only do recognition every Nth frame

face_locations = []
face_names = []

frame_count = 0
start_time = time.time()
fps = 0

# ============ FUNCTIONS ============

def recognize_faces(frame):
    """
    Run face detection + recognition on a single frame.
    This is the heavy part and we call it only every Nth frame.
    """
    global face_locations, face_names

    # Resize the frame to speed up face detection/encoding
    small_frame = cv2.resize(frame, (0, 0), fx=1.0 / cv_scaler, fy=1.0 / cv_scaler)

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations (HOG is default; fast and CPU-friendly)
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Compute encodings (use the SMALL model for speed)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations, model="small"
    )

    face_names_local = []
    for face_encoding in face_encodings:
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names_local.append(name)

    face_names[:] = face_names_local  # update global list


def draw_results(frame):
    """
    Draw boxes and labels for the last known face locations & names.
    """
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 2)

        # Draw a label above the face
        cv2.rectangle(frame, (left - 3, top - 30), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 5, top - 7), font, 0.7, (255, 255, 255), 1)

    return frame


def calculate_fps():
    """
    Simple FPS calculator updated once per second.
    """
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


# ============ MAIN LOOP ============

frame_index = 0

while True:
    # Capture a frame
    frame = picam2.capture_array()

    # Only run heavy face recognition every Nth frame
    if frame_index % process_every_n_frames == 0:
        recognize_faces(frame)

    frame_index += 1

    # Draw results from the last recognition step
    display_frame = draw_results(frame)

    # FPS overlay
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

    # Show the frame
    cv2.imshow("Video", display_frame)

    # Exit on 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
