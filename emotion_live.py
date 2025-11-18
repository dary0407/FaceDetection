import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "face_landmarker.task"

BaseOptions = python.BaseOptions
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarker = vision.FaceLandmarker
RunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    output_face_blendshapes=True,
    num_faces=1,
)

detector = FaceLandmarker.create_from_options(options)

# mapping
def guess_emotion(blendshapes):
    scores = {b.category_name: b.score for b in blendshapes}

    # câțiva mușchi relevanți
    happy_score = scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)
    sad_score = scores.get("mouthFrownLeft", 0) + scores.get("mouthFrownRight", 0)
    surprised_score = scores.get("jawOpen", 0) + scores.get("eyeWideLeft", 0) + scores.get("eyeWideRight", 0)
    angry_score = scores.get("browDownLeft", 0) + scores.get("browDownRight", 0)

    vals = {
        "HAPPY": happy_score,
        "SAD": sad_score,
        "SURPRISED": surprised_score,
        "ANGRY": angry_score,
        "NEUTRAL": scores.get("neutral", 0.1),
    }

    emo = max(vals, key=vals.get)
    return emo, vals[emo]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # mp.Image funcționează cu mediapipe 0.10.11
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.face_blendshapes:
        blends = result.face_blendshapes[0]
        emo, val = guess_emotion(blends)

        text = f"{emo} ({val:.2f})"
        cv2.putText(frame, text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("AI Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
