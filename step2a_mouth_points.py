import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# mouth landmarks
TOP_MOUTH = 13
BOTTOM_MOUTH = 14

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

            # TOP and BOTTOM coords for mouth
            lm = face.landmark
            top = lm[TOP_MOUTH]
            bottom = lm[BOTTOM_MOUTH]

            # Transform to pixel coords
            x_top, y_top = int(top.x * w), int(top.y * h)
            x_bottom, y_bottom = int(bottom.x * w), int(bottom.y * h)

            # Draw points
            cv2.circle(frame, (x_top, y_top), 4, (0,255,0), -1)
            cv2.circle(frame, (x_bottom, y_bottom), 4, (0,0,255), -1)

            # Show points
            cv2.putText(frame, f"TOP: {y_top}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(frame, f"BOTTOM: {y_bottom}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        cv2.imshow("Step 2A - Mouth Points", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
