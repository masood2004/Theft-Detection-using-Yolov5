import torch
import cv2 as cv
import face_recognition as fc
import os
import yagmail
from datetime import datetime
import warnings
import pygame
import numpy as np
from dotenv import load_dotenv

# Initialize environment and suppress warnings
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)
pygame.mixer.init()

# Configuration
KNOWN_FACES_DIR = 'known_faces'
SUSPECTS_FILE = 'known_suspects.dat'
ALARM_SOUND = "alarm.mp3"
RECORDING_FPS = 20.0
FRAME_SIZE = (640, 480)
SECURITY_EMAIL = os.getenv("SECURITY_EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
ALARM_COOLDOWN = 3  # Seconds to wait before stopping alarm

# Global states
alarm_playing = False
recording = False
video_writer = None
last_unknown_detection = None
known_suspects = []

# Initialize email client
yag = yagmail.SMTP(user=SECURITY_EMAIL, password=EMAIL_PASSWORD)

# Create directories if missing
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


def load_known_faces():
    encodings = []
    names = []
    try:
        for file in os.listdir(KNOWN_FACES_DIR):
            image = fc.load_image_file(f"{KNOWN_FACES_DIR}/{file}")
            encoding = fc.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                names.append(os.path.splitext(file)[0])
    except Exception as e:
        print(f"Error loading known faces: {e}")
    return encodings, names


def load_suspects():
    suspects = []
    try:
        with open(SUSPECTS_FILE, "rb") as f:
            while True:
                encoding = np.load(f, allow_pickle=True)
                name = f.readline().decode().strip()
                suspects.append((encoding, name))
    except Exception as e:
        print(f"Error loading suspects: {e}")
    return suspects


def save_suspect(encoding, name):
    try:
        with open(SUSPECTS_FILE, "ab") as f:
            np.save(f, encoding)
            f.write(f"{name}\n".encode())
    except Exception as e:
        print(f"Error saving suspect: {e}")


def play_alarm():
    global alarm_playing
    if not alarm_playing:
        pygame.mixer.music.load(ALARM_SOUND)
        pygame.mixer.music.play(-1)
        alarm_playing = True


def stop_alarm():
    global alarm_playing
    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False


def start_recording():
    global recording, video_writer
    if not recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"surveillance_{timestamp}.avi"
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        video_writer = cv.VideoWriter(
            filename, fourcc, RECORDING_FPS, FRAME_SIZE)
        recording = True
        print(f"[RECORDING] Started continuous recording: {filename}")


def stop_recording():
    global recording, video_writer
    if video_writer is not None:
        video_writer.release()
        print("[RECORDING] Stopped continuous recording")
    recording = False


def send_alert(image_path):
    try:
        subject = "Suspicious Activity Detected!"
        body = "A suspicious activity was detected. See attached image."
        yag.send(to=SECURITY_EMAIL, subject=subject,
                 contents=body, attachments=image_path)
    except Exception as e:
        print(f"Error sending email: {e}")


def process_frame(frame, model, known_face_encodings, known_face_names):
    global last_unknown_detection
    current_time = datetime.now()
    unknown_in_frame = False

    # Start recording on first frame processing
    start_recording()

    # Motion detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)

    if 'avg' not in process_frame.__dict__:
        process_frame.avg = gray.copy().astype("float")

    cv.accumulateWeighted(gray, process_frame.avg, 0.5)
    frame_delta = cv.absdiff(gray, cv.convertScaleAbs(process_frame.avg))
    thresh = cv.threshold(frame_delta, 5, 255, cv.THRESH_BINARY)[1]

    if np.sum(thresh) > 10000:
        results = model(frame)
        detections = results.pandas().xyxy[0]

        for _, row in detections.iterrows():
            if row['name'] == 'person' and row['confidence'] > 0.6:
                x1, y1, x2, y2 = map(
                    int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                rgb_face = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
                face_locations = fc.face_locations(rgb_face)

                if not face_locations:
                    continue

                face_encodings = fc.face_encodings(rgb_face, face_locations)

                if face_encodings:
                    suspect_matches = fc.compare_faces(
                        [s[0] for s in known_suspects], face_encodings[0])
                    name = "Unknown"

                    if True in suspect_matches:
                        name = known_suspects[suspect_matches.index(True)][1]
                        play_alarm()
                        unknown_in_frame = True
                    else:
                        authorized_matches = fc.compare_faces(
                            known_face_encodings, face_encodings[0])

                        if True in authorized_matches:
                            name = known_face_names[authorized_matches.index(
                                True)]
                            stop_alarm()
                        else:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path = f"suspect_{timestamp}.jpg"
                            cv.imwrite(img_path, frame)
                            save_suspect(
                                face_encodings[0], f"suspect_{timestamp}")
                            known_suspects.append(
                                (face_encodings[0], f"suspect_{timestamp}"))
                            send_alert(img_path)
                            play_alarm()
                            unknown_in_frame = True
                            last_unknown_detection = current_time

                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv.putText(frame, name, (x1, y1-10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Auto-stop alarm logic
    if unknown_in_frame:
        last_unknown_detection = current_time
    elif last_unknown_detection and (current_time - last_unknown_detection).seconds > ALARM_COOLDOWN:
        stop_alarm()
        last_unknown_detection = None

    return frame


def main():
    try:
        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s', pretrained=True)
    except Exception as e:
        print(f"Model error: {e}")
        return

    known_face_encodings, known_face_names = load_known_faces()
    global known_suspects
    known_suspects = load_suspects()

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    print("[SYSTEM ACTIVE] Security monitoring started...")
    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv.resize(frame, FRAME_SIZE)
                processed_frame = process_frame(
                    frame, model, known_face_encodings, known_face_names)

                if recording:
                    video_writer.write(processed_frame)

                cv.imshow("Security Monitor", processed_frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            except KeyboardInterrupt:
                raise
    except KeyboardInterrupt:
        print("\n[INFO] Stopping surveillance...")
    finally:
        stop_alarm()
        stop_recording()
        cap.release()
        cv.destroyAllWindows()
        print("[SYSTEM SHUTDOWN]")


if __name__ == "__main__":
    main()
