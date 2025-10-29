"""
emotion_recognition.py
Real-time facial detection using OpenCV with basic emotion estimation.

Usage:
    python emotion_recognition.py

Notes:
 - Make sure your virtual environment is activated and required packages are installed.
 - Press 'q' to quit the program.
 - Press 'r' to reset emotion detection.
"""

import cv2
import time
import numpy as np
from collections import deque

# ---------- Config ----------
# Face detection settings
HAAR_PATH = "assets/haarcascades/haarcascade_frontalface_default.xml"
MIN_FACE_SIZE = (60, 60)
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5

# Performance settings
MOVING_AVG_SIZE = 30  # Number of frames to average FPS
MAX_FACES = 4  # Maximum number of faces to track

# Simple emotion detection based on face geometry
EMOTIONS = ['neutral', 'happy', 'surprised', 'sad']

def load_face_detector(haar_path=HAAR_PATH):
    """Load OpenCV's Haar cascade face detector."""
    try:
        if haar_path:
            face_cascade = cv2.CascadeClassifier(haar_path)
            if face_cascade.empty():
                raise Exception("Failed to load Haar cascade from provided path.")
            return face_cascade
    except Exception as e:
        print(f"[INFO] Could not load local Haar cascade: {e}")

    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if face_cascade.empty():
            raise Exception("OpenCV default Haar cascade failed to load.")
        return face_cascade
    except Exception as e:
        print(f"[ERROR] No Haar cascade available: {e}")
        return None

class FaceTracker:
    def __init__(self):
        self.emotion_history = deque(maxlen=10)
        self.last_emotion = 'neutral'
        self.confidence = 0
        self.face_roi = None
    
    def update(self, face_roi):
        """Update face tracker with new ROI and estimate emotion."""
        self.face_roi = face_roi
        
        # Extract face features
        h, w = face_roi.shape[:2]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Simple emotion estimation based on face geometry
        try:
            # Detect eyes
            eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eyes_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(eyes) >= 2:
                # Calculate eye separation ratio
                eye_distance = abs(eyes[0][0] - eyes[1][0])
                eye_ratio = eye_distance / w
                
                # Estimate emotion based on eye ratio
                if eye_ratio > 0.3:  # Wide eyes
                    emotion = 'surprised'
                    conf = 65 + (eye_ratio - 0.3) * 100
                elif eye_ratio < 0.2:  # Narrow eyes
                    emotion = 'happy'
                    conf = 60 + (0.2 - eye_ratio) * 100
                else:
                    emotion = 'neutral'
                    conf = 50
                
                self.emotion_history.append(emotion)
                
                # Use most common emotion from history
                if self.emotion_history:
                    emotions = list(self.emotion_history)
                    most_common = max(set(emotions), key=emotions.count)
                    self.last_emotion = most_common
                    self.confidence = conf
            
            return self.last_emotion, min(100, max(0, self.confidence))
            
        except Exception as e:
            print(f"[WARN] Emotion estimation failed: {e}")
            return self.last_emotion, 40

def main():
    # Initialize face detector
    face_detector = load_face_detector()
    if face_detector is None:
        print("[FATAL] No face detector available. Exiting.")
        return

    # Try different camera indices if default (0) doesn't work
    camera_found = False
    for camera_index in range(4):  # Try indices 0-3
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"[INFO] Found camera at index {camera_index}")
            camera_found = True
            break
        cap.release()

    if not camera_found:
        print("[ERROR] Could not find any working webcam. Check camera connections and permissions.")
        return

    # Initialize FPS calculation and face trackers
    frame_times = []
    face_trackers = {}
    
    print("[INFO] Starting webcam. Press 'q' to quit, 'r' to reset tracking.")

    while True:
        start_time = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame not received. Exiting loop.")
            break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar cascade
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE
        )

        # Limit number of faces to track
        faces = faces[:MAX_FACES] if len(faces) > MAX_FACES else faces
        
        # Track and analyze each face
        for i, (x, y, w, h) in enumerate(faces):
            # Expand box for better face detection
            pad = int(0.1 * w)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            # Get or create face tracker
            if i not in face_trackers:
                face_trackers[i] = FaceTracker()
            
            # Extract and analyze face ROI
            face_roi = frame[y1:y2, x1:x2]
            emotion, confidence = face_trackers[i].update(face_roi)

            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create emotion label
            label = f"{emotion} ({confidence:.1f}%)"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Calculate and display FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        if len(frame_times) > MOVING_AVG_SIZE:
            frame_times.pop(0)
        
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 10), 2)

        # Show instructions
        cv2.putText(display_frame, "Press 'q' to quit, 'r' to reset", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 255, 10), 2)

        # Show the frame
        cv2.imshow("Real-time Emotion Recognition", display_frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("[INFO] Resetting face trackers")
            face_trackers.clear()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program terminated.")

if __name__ == "__main__":
    main()