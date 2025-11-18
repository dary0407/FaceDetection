import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

TOP_MOUTH = 13
BOTTOM_MOUTH = 14

def distance(y1, y2):
    return abs(y1 - y2)

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

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            lm = face.landmark

            # Pixel coords
            y_top  = int(lm[TOP_MOUTH].y * h)     
            y_bottom = int(lm[BOTTOM_MOUTH].y * h)

            # Calculate „mouth_open”
            mouth_open = distance(y_top, y_bottom)

            # Draw points
            cv2.circle(frame, (int(lm[TOP_MOUTH].x*w), y_top), 4, (0,255,0), -1)
            cv2.circle(frame, (int(lm[BOTTOM_MOUTH].x*w), y_bottom), 4, (0,0,255), -1)

            # Show value
            cv2.putText(
                frame,
                f"mouth_open = {mouth_open}",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,0),
                2
            )

        cv2.imshow("Step 2B - Mouth Open", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
