import tkinter as tk
from tkinter import scrolledtext


class ChatUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AI Mood Chat")
        self.window.geometry("1000x600")

        #LAYOUT
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill="both", expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True)

        # camera & emotions frame
        self.camera_label = tk.Label(left_frame, bg="black")
        self.camera_label.pack(fill="both", expand=True, padx=10, pady=10)

        self.emotion_label = tk.Label(
            left_frame,
            text="Emotion: ---",
            font=("Calibri", 12),
            fg="gray"
        )
        self.emotion_label.pack(pady=5)

        # chat area
        self.chat_area = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=("Calibri", 12)
        )
        self.chat_area.pack(expand=True, fill="both", padx=10, pady=(10, 0))

        bottom_frame = tk.Frame(right_frame)
        bottom_frame.pack(fill="x", padx=10, pady=10)

        self.entry = tk.Entry(bottom_frame, font=("Calibri", 12))
        self.entry.pack(side="left", fill="x", expand=True)

        send_button = tk.Button(bottom_frame, text="Send", command=self.send_message)
        send_button.pack(side="right", padx=(10, 0))

        self.entry.bind("<Return>", self.send_message_event)

        # logic
        self.ai_callback = None
        self.current_emotion = "NEUTRAL"
        self._camera_image = None  # ca să nu fie garbage collected
    def update_camera(self, tk_image):
        self._camera_image = tk_image
        self.camera_label.config(image=self._camera_image)

    def update_emotion(self, emo):
        self.emotion_label.config(text=f"Emotion: {emo}")

    # chat logic
    def send_message_event(self, event):
        self.send_message()

    def send_message(self):
        text = self.entry.get().strip()
        if not text:
            return

        self.chat_area.insert(tk.END, f"You: {text}\n")
        self.chat_area.see(tk.END)
        self.entry.delete(0, tk.END)

        if self.ai_callback:
            try:
                reply = self.ai_callback(text)
            except Exception as e:
                reply = f"(Eroare AI: {e})"

            if reply:
                self.chat_area.insert(tk.END, f"AI: {reply}\n")
                self.chat_area.see(tk.END)

    def run(self):
        self.window.mainloop()
