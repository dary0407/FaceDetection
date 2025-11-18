import cv2
from PIL import Image, ImageTk

from emotion_chat import detect_emotion
from chat_ui import ChatUI
from chat_ai_groq import groq_reply


def run():
    ui = ChatUI()
    cap = cv2.VideoCapture(0)

    def update_camera():
        ret, frame = cap.read()
        if not ret:
            ui.window.after(30, update_camera)
            return

        # mirroring image
        frame_bgr = cv2.flip(frame, 1)

        # emotions
        emo = detect_emotion(frame_bgr)
        ui.current_emotion = emo
        ui.update_emotion(emo)

        # convert to RGB for Tkinter
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        img = img.resize((480, 360))

        tk_img = ImageTk.PhotoImage(img)
        ui.update_camera(tk_img)

        ui.window.after(30, update_camera)

    ui.ai_callback = lambda user_text: groq_reply(
        user_text,
        ui.current_emotion
    )

    update_camera()

    try:
        ui.run()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
