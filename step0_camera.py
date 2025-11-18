import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Can't open camera :(")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read camera frame")
        break

    # oglindă
    frame = cv2.flip(frame, 1)

    cv2.imshow("Step 0 - Camera", frame)

    # quit with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
