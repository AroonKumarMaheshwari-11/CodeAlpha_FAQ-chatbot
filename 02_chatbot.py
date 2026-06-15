import nltk
import string
import threading
import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile, os
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords



faqs = [
    {"question": "hi",
     "answer": "Hi! I am FAQ-Chatbot AI. How can I help you?"},
    {"question": "hello",
     "answer": "Hello! Welcome to the AI FAQ Chatbot. Ask me anything about AI!"},
    {"question": "can you tell about some artificial intelligence terms?",
     "answer": "Yes! Sure, ask me freely about AI terms."},
    {"question": "What is Artificial Intelligence?",
     "answer": "Artificial Intelligence (AI) is the simulation of human intelligence by machines to perform tasks like reasoning, learning, and problem-solving."},
    {"question": "What is Machine Learning?",
     "answer": "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed."},
    {"question": "What is Deep Learning?",
     "answer": "Deep Learning is a subset of Machine Learning that uses multi-layered neural networks to analyze large amounts of data."},
    {"question": "What is a Neural Network?",
     "answer": "A Neural Network is a series of algorithms that mimic the human brain to recognize patterns and solve complex problems."},
    {"question": "What is Natural Language Processing?",
     "answer": "NLP (Natural Language Processing) is a branch of AI that helps computers understand, interpret, and generate human language."},
    {"question": "What is a chatbot?",
     "answer": "A chatbot is an AI-powered program designed to simulate conversation with users, typically used for customer service or FAQs."},
    {"question": "What is Python?",
     "answer": "Python is a high-level, easy-to-learn programming language widely used in AI, Machine Learning, and data science."},
    {"question": "What is data science?",
     "answer": "Data Science is a field that uses statistics, programming, and machine learning to extract insights from data."},
    {"question": "What is supervised learning?",
     "answer": "Supervised Learning is a type of ML where the model is trained on labeled data to predict outcomes for new data."},
    {"question": "What is unsupervised learning?",
     "answer": "Unsupervised Learning is a type of ML where the model finds hidden patterns in data without labeled responses."},
    {"question": "What is a large language model?",
     "answer": "A Large Language Model (LLM) is an AI model trained on massive text data to understand and generate human-like text. Examples: GPT, Claude."},
    {"question": "What is computer vision?",
     "answer": "Computer Vision is a field of AI that enables computers to interpret and understand visual information from images and videos."},
    {"question": "What is reinforcement learning?",
     "answer": "Reinforcement Learning is a type of ML where an agent learns by interacting with an environment and receiving rewards or penalties."},
    {"question": "What is overfitting?",
     "answer": "Overfitting occurs when a model learns the training data too well and performs poorly on new, unseen data."},
    {"question": "What is a dataset?",
     "answer": "A dataset is a collection of organized data used to train, test, and evaluate machine learning models."},
    {"question": "bye",
     "answer": "Goodbye! Have a great day! 👋"},
    {"question": "thank you",
     "answer": "You're welcome! Feel free to ask more questions anytime. 😊"},
]


stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

faq_questions = [preprocess(f["question"]) for f in faqs]
vectorizer    = TfidfVectorizer()
tfidf_matrix  = vectorizer.fit_transform(faq_questions)

def get_answer(user_input):
    cleaned = preprocess(user_input)
    if not cleaned.strip():
        return "Please type or speak a valid question."
    user_vec   = vectorizer.transform([cleaned])
    scores     = cosine_similarity(user_vec, tfidf_matrix)
    best_idx   = scores.argmax()
    best_score = scores[0][best_idx]
    if best_score < 0.2:
        return "Sorry, I don't have an answer for that. Please try rephrasing your question."
    return faqs[best_idx]["answer"]

# ─── TTS Engine ───────────────────────────────────────────────────────────────
def speak(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()

BG_DARK    = "#0f0f1a"
BG_PANEL   = "#1a1a2e"
BG_INPUT   = "#16213e"
ACCENT     = "#7c3aed"
ACCENT2    = "#06b6d4"
TEXT_WHITE = "#f1f5f9"
TEXT_MUTED = "#94a3b8"
MIC_ACTIVE = "#ef4444"
MIC_IDLE   = "#06b6d4"


class FAQChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 AI FAQ Chatbot — CodeAlpha")
        self.root.geometry("780x650")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)
        self.listening = False
        self._build_ui()
        self._welcome()

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG_PANEL, pady=14)
        header.pack(fill="x")
        tk.Label(header, text="🤖", font=("Segoe UI Emoji", 26),
                 bg=BG_PANEL, fg=ACCENT).pack(side="left", padx=(20, 6))
        title_frame = tk.Frame(header, bg=BG_PANEL)
        title_frame.pack(side="left")
        tk.Label(title_frame, text="AI FAQ Chatbot",
                 font=("Segoe UI", 17, "bold"),
                 bg=BG_PANEL, fg=TEXT_WHITE).pack(anchor="w")
        tk.Label(title_frame, text="Powered by NLP · TF-IDF · Cosine Similarity",
                 font=("Segoe UI", 9), bg=BG_PANEL, fg=TEXT_MUTED).pack(anchor="w")
        self.status_dot = tk.Label(header, text="● Online",
                                   font=("Segoe UI", 9, "bold"),
                                   bg=BG_PANEL, fg="#22c55e")
        self.status_dot.pack(side="right", padx=20)
        tk.Frame(self.root, bg=ACCENT, height=2).pack(fill="x")


        self.chat_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state="disabled",
            bg=BG_DARK, fg=TEXT_WHITE,
            font=("Segoe UI", 11), bd=0, relief="flat",
            padx=16, pady=12, insertbackground=TEXT_WHITE)
        self.chat_area.pack(fill="both", expand=True)
        self.chat_area.tag_config("user_label", foreground=ACCENT,
                                  font=("Segoe UI", 9, "bold"))
        self.chat_area.tag_config("user_msg",   foreground=TEXT_WHITE,
                                  font=("Segoe UI", 11), lmargin1=20, lmargin2=20)
        self.chat_area.tag_config("bot_label",  foreground=ACCENT2,
                                  font=("Segoe UI", 9, "bold"))
        self.chat_area.tag_config("bot_msg",    foreground=TEXT_WHITE,
                                  font=("Segoe UI", 11), lmargin1=20, lmargin2=20)
        self.chat_area.tag_config("muted",      foreground=TEXT_MUTED,
                                  font=("Segoe UI", 9, "italic"))

        tk.Frame(self.root, bg="#2d2d4e", height=1).pack(fill="x")

        # Input bar
        input_bar = tk.Frame(self.root, bg=BG_INPUT, pady=12)
        input_bar.pack(fill="x")
        self.mic_btn = tk.Button(
            input_bar, text="🎤", font=("Segoe UI Emoji", 16),
            bg=MIC_IDLE, fg="white", relief="flat",
            activebackground=MIC_ACTIVE, activeforeground="white",
            cursor="hand2", width=3, bd=0, command=self._toggle_voice)
        self.mic_btn.pack(side="left", padx=(14, 6))
        self.entry = tk.Entry(
            input_bar, font=("Segoe UI", 12),
            bg="#0f172a", fg=TEXT_WHITE,
            insertbackground=TEXT_WHITE, relief="flat", bd=0,
            highlightthickness=1, highlightcolor=ACCENT,
            highlightbackground="#2d2d4e")
        self.entry.pack(side="left", fill="x", expand=True, ipady=10, padx=6)
        self.entry.bind("<Return>", lambda e: self._send_text())
        self.entry.focus()
        tk.Button(input_bar, text="Send ➤",
                  font=("Segoe UI", 11, "bold"),
                  bg=ACCENT, fg="white",
                  activebackground="#6d28d9", activeforeground="white",
                  relief="flat", bd=0, padx=18, pady=8,
                  cursor="hand2", command=self._send_text
                  ).pack(side="left", padx=(6, 14))


        footer = tk.Frame(self.root, bg=BG_DARK, pady=6)
        footer.pack(fill="x")
        tk.Label(footer,
                 text="CodeAlpha Internship Project  |  Python · NLTK · sklearn · Tkinter · sounddevice · pyttsx3",
                 font=("Segoe UI", 8), bg=BG_DARK, fg=TEXT_MUTED).pack()

    def _welcome(self):
        self._append_bot("👋 Welcome! I'm your AI FAQ Chatbot.\n"
                         "Ask me anything about Artificial Intelligence.\n"
                         "You can type OR click 🎤 to speak!")

    def _append_user(self, text):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", "\n  You\n", "user_label")
        self.chat_area.insert("end", f"  {text}\n", "user_msg")
        self.chat_area.config(state="disabled")
        self.chat_area.see("end")

    def _append_bot(self, text):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", "\n  🤖 Bot\n", "bot_label")
        self.chat_area.insert("end", f"  {text}\n", "bot_msg")
        self.chat_area.config(state="disabled")
        self.chat_area.see("end")

    def _append_muted(self, text):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", f"\n  {text}\n", "muted")
        self.chat_area.config(state="disabled")
        self.chat_area.see("end")

    def _send_text(self):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.entry.delete(0, "end")
        self._process_input(user_input)

    def _process_input(self, user_input):
        self._append_user(user_input)
        answer = get_answer(user_input)
        self._append_bot(answer)
        speak(answer)

    def _toggle_voice(self):
        if self.listening:
            return
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        self.listening = True
        self.mic_btn.config(bg=MIC_ACTIVE, text="⏹")
        self.status_dot.config(text="● Listening…", fg="#facc15")
        self._append_muted("🎤 Listening… speak now")

        recognizer = sr.Recognizer()
        tmp_path   = None
        try:
            # Record with sounddevice
            sample_rate = 16000
            duration    = 7
            recording   = sd.rec(int(duration * sample_rate),
                                  samplerate=sample_rate,
                                  channels=1, dtype='int16')
            sd.wait()

            # ✅ Close file handle BEFORE writing — fixes WinError 32
            tmp      = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            wav.write(tmp_path, sample_rate, recording)

            # Recognize
            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)

            text = recognizer.recognize_google(audio)
            if text.strip():
                self.root.after(0, self._process_input, text)
            else:
                self.root.after(0, self._append_muted, "⚠ No speech detected. Try again.")

        except sr.UnknownValueError:
            self.root.after(0, self._append_muted, "⚠ Could not understand. Please speak clearly.")
        except sr.RequestError:
            self.root.after(0, self._append_muted, "⚠ Internet needed for voice recognition.")
        except Exception as ex:
            self.root.after(0, self._append_muted, f"⚠ Error: {ex}")
        finally:
            # ✅ Safe cleanup
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            self.listening = False
            self.root.after(0, self.mic_btn.config,    {"bg": MIC_IDLE,   "text": "🎤"})
            self.root.after(0, self.status_dot.config, {"text": "● Online", "fg": "#22c55e"})



if __name__ == "__main__":
    root = tk.Tk()
    app  = FAQChatbotApp(root)
    root.mainloop()
