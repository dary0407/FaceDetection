import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
TOP_MOUTH = 13
BOTTOM_MOUTH = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
BROW_LEFT = 70
BROW_RIGHT = 300
NOSE_TOP = 6

def dist(a, b):
    return abs(a - b)

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        emotion = "neutral"

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            lm = face.landmark

            def px(id):
                return int(lm[id].x * w), int(lm[id].y * h)

            # METRICS
            # mouth
            _, y_top  = px(TOP_MOUTH)
            _, y_bottom = px(BOTTOM_MOUTH)
            mouth_open = dist(y_top, y_bottom)

            # smile
            x_left, _  = px(LEFT_MOUTH)
            x_right, _ = px(RIGHT_MOUTH)
            smile = dist(x_left, x_right)

            # eyes
            _, y_lt = px(LEFT_EYE_TOP)
            _, y_lb = px(LEFT_EYE_BOTTOM)
            _, y_rt = px(RIGHT_EYE_TOP)
            _, y_rb = px(RIGHT_EYE_BOTTOM)
            eyes = (dist(y_lt, y_lb) + dist(y_rt, y_rb)) / 2

            # eyebrows
            _, y_bl = px(BROW_LEFT)
            _, y_br = px(BROW_RIGHT)
            _, y_nose = px(NOSE_TOP)
            brows = (dist(y_bl, y_nose) + dist(y_br, y_nose)) / 2

            # EMOTION RULES

            # Surprised
            if mouth_open > 35 and eyes > 14 and brows > 18:
                emotion = "surprised"

            # Happy
            elif smile > 120:
                emotion = "happy"

            # Angry
            elif brows < 12 and mouth_open < 25:
                emotion = "angry"

            # Sad
            elif smile < 80 and brows > 17:
                emotion = "sad"

            # Neutral is default


            # DRAW EMOTION LABEL
            cv2.putText(frame, f"Emotion: {emotion}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow("Step 3B - Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
