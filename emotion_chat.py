import time
import random
import cv2
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

last_emo = "NEUTRAL"


def guess_emotion(blendshapes):
    scores = {b.category_name: b.score for b in blendshapes}

    happy = scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)
    sad = scores.get("mouthFrownLeft", 0) + scores.get("mouthFrownRight", 0)
    surprised = scores.get("jawOpen", 0) + scores.get("eyeWideLeft", 0) + scores.get("eyeWideRight", 0)
    angry = scores.get("browDownLeft", 0) + scores.get("browDownRight", 0)

    vals = {
        "HAPPY": happy,
        "SAD": sad,
        "SURPRISED": surprised,
        "ANGRY": angry,
        "NEUTRAL": scores.get("neutral", 0.1),
    }

    emo = max(vals, key=vals.get)
    return emo


def detect_emotion(frame):
    global last_emo

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.face_blendshapes:
        blends = result.face_blendshapes[0]
        last_emo = guess_emotion(blends)

    return last_emo
