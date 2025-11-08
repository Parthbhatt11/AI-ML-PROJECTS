import tkinter as tk
from tkinter import messagebox
import threading
import pyttsx3
from predictor import predict_disease
from chat_parser import extract_symptoms

# ===================================================
# 🎤 Text-to-Speech Engine
# ===================================================
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    """Safely run text-to-speech in a separate thread."""
    def _speak():
        try:
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            pass
        except Exception:
            pass

    threading.Thread(target=_speak, daemon=True).start()


# ===================================================
# 🎨 UI DESIGN CONFIG
# ===================================================
BG_COLOR = "#F4F6F7"
HEADER_COLOR = "#0078D7"
BTN_COLOR = "#0078D7"
FONT_MAIN = ("Segoe UI", 11)
FONT_TITLE = ("Segoe UI Semibold", 18)
TEXT_COLOR = "#222"

# Confidence color mapping (matches predictor)
CONFIDENCE_COLORS = {
    "High": "#28a745",       # Green
    "Moderate": "#ffc107",   # Yellow
    "Low": "#dc3545"         # Red
}


# ===================================================
# 🩺 Main Application
# ===================================================
class SymptomCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 AI Symptom Checker — Multi-Disease Mode")
        self.root.geometry("820x680")
        self.root.configure(bg=BG_COLOR)

        # Header
        header = tk.Canvas(root, height=60, bg=HEADER_COLOR, highlightthickness=0)
        header.pack(fill="x")
        header.create_text(30, 30, text="AI Symptom Checker", font=FONT_TITLE, fill="white", anchor="w")

        # Input Section
        frame = tk.Frame(root, bg=BG_COLOR)
        frame.pack(pady=20)

        tk.Label(
            frame,
            text="Enter your symptoms (comma-separated):",
            font=FONT_MAIN, bg=BG_COLOR, fg=TEXT_COLOR
        ).pack(pady=5)

        self.symptom_entry = tk.Text(frame, height=4, width=75, font=FONT_MAIN, wrap="word")
        self.symptom_entry.pack(pady=10)

        tk.Button(
            frame, text="🔍 Analyze Symptoms", font=("Segoe UI", 11, "bold"),
            bg=BTN_COLOR, fg="white", relief="flat", padx=10, pady=6,
            command=self.start_analysis
        ).pack(pady=10)

        # Results Section
        tk.Label(
            root, text="🧾 Possible Diagnoses:",
            font=("Segoe UI Semibold", 13), bg=BG_COLOR, fg=TEXT_COLOR
        ).pack(pady=(20, 5))

        self.result_box = tk.Text(
            root, height=20, width=95, font=FONT_MAIN, wrap="word",
            state="disabled", bg="white", relief="flat"
        )
        self.result_box.pack(padx=20, pady=5)

        # Configure tags for colors and formatting
        self.result_box.tag_configure("high", foreground="#28a745", font=("Segoe UI Semibold", 11))
        self.result_box.tag_configure("moderate", foreground="#cba300", font=("Segoe UI", 11))
        self.result_box.tag_configure("low", foreground="#dc3545", font=("Segoe UI", 11))
        self.result_box.tag_configure("title", font=("Segoe UI Semibold", 12), spacing1=4, spacing3=2)
        self.result_box.tag_configure("divider", foreground="#666", spacing1=6, spacing3=6)

    # ===================================================
    # Core Logic
    # ===================================================
    def start_analysis(self):
        """Run the analysis safely in a thread."""
        threading.Thread(target=self.analyze_symptoms, daemon=True).start()

    def analyze_symptoms(self):
        """Extract symptoms, run prediction, and show results."""
        user_input = self.symptom_entry.get("1.0", "end").strip()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter your symptoms first.")
            return

        # Extract symptoms
        symptoms = extract_symptoms(user_input)
        if not symptoms:
            symptoms = [s.strip() for s in user_input.split(",") if s.strip()]

        if len(symptoms) < 2:
            messagebox.showinfo("Info", "Please enter at least two symptoms for accurate prediction.")
            return

        self.show_result("Analyzing your symptoms... Please wait ⏳")

        try:
            results = predict_disease(symptoms)
            if not results or "error" in results[0]:
                error_msg = results[0].get("error", "Prediction failed. Please check your model or data.")
                self.show_result(f"⚠️ {error_msg}")
                speak("Sorry, I couldn't process your symptoms correctly.")
                return

            self.result_box.config(state="normal")
            self.result_box.delete("1.0", "end")

            # Display results dynamically
            for i, res in enumerate(results, 1):
                disease = res['disease']
                score = res['score']
                desc = res['description']
                conf = res['confidence']
                precautions = ', '.join(res.get('precautions', [])[:4]) or "No specific precautions."

                # Determine tag color
                tag = conf.lower()
                color_tag = tag if tag in ["high", "moderate", "low"] else "moderate"

                # Title Line
                self.result_box.insert(tk.END, f"🔹 {i}. {disease} ({score}% — {conf} confidence)\n", color_tag)

                # Description
                self.result_box.insert(tk.END, f"📖 {desc}\n", "title")

                # Precautions
                self.result_box.insert(tk.END, f"💡 Precautions: {precautions}\n", "title")

                # Divider between entries
                if i != len(results):
                    self.result_box.insert(tk.END, "──────────────────────────────\n", "divider")

            self.result_box.config(state="disabled")

            # Voice feedback for top disease
            top_name = results[0]['disease']
            top_score = results[0]['score']
            top_conf = results[0]['confidence']
            speak(f"Top possible condition: {top_name}. Confidence level {top_conf}, {int(top_score)} percent.")

        except Exception as e:
            self.show_result(f"❌ Error during analysis:\n{e}")
            speak("An error occurred while analyzing symptoms.")

    # ===================================================
    # Output Display
    # ===================================================
    def show_result(self, text):
        """Safely update result box."""
        self.result_box.config(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", text)
        self.result_box.config(state="disabled")
        self.result_box.yview(tk.END)


# ===================================================
# 🚀 Launch App
# ===================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SymptomCheckerApp(root)
    root.mainloop()
