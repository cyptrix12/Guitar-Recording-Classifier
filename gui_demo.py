import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import librosa
import numpy as np
import joblib

MODEL_FILES = {
    "KNN": "trained_model/knn_model.pkl",
    "SVM": "trained_model/svm_model.pkl",
    "FNN": "trained_model/ffn_model.pkl",
    "NN" : "trained_model/cnn_model.pkl",
}
SCALER_PATH       = "trained_model/scaler.pkl"
LABELENCODER_PATH = "trained_model/label_encoder.pkl"

SAMPLE_RATE = 48000
N_MFCC      = 42

class GuitarClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("GADA demo")
        master.resizable(False, False)

        # —––––– STYLING –––––—
        style = ttk.Style(master)
        style.theme_use("clam")  
        style.configure("TButton", font=("Segoe UI", 11), padding=8)
        style.configure("TLabel", font=("Segoe UI", 11))
        style.configure("Header.TButton", font=("Segoe UI", 13, "bold"))
        style.configure("Model.TLabel", font=("Segoe UI", 11, "bold"))

        # —––––– FRAMES –––––—
        top_frame    = ttk.Frame(master, padding=10)
        results_frame= ttk.Frame(master, padding=(10,0,10,10))
        top_frame.grid   (row=0, column=0, sticky="ew")
        results_frame.grid(row=1, column=0, sticky="nsew")

        # —––––– BUTTON –––––—
        self.btn_open = ttk.Button(
            top_frame, text="Select WAV File…", style="Header.TButton",
            command=self.open_file
        )
        self.btn_open.pack(fill="x")

        self.file_label = ttk.Label(top_frame, text="No file selected", font=("Segoe UI", 10, "italic"))
        self.file_label.pack(fill="x", pady=(5,10))

        # —––––– DYNAMIC RESULTS --
        self.model_labels    = {}
        self.progress_bars   = {}
        for i, name in enumerate(MODEL_FILES):
            lbl = ttk.Label(results_frame, text=f"{name}:", style="Model.TLabel")
            lbl.grid(row=i, column=0, sticky="w", pady=4)

            val = ttk.Label(results_frame, text="—", width=16)
            val.grid(row=i, column=1, sticky="w", padx=(5,20))

            pb  = ttk.Progressbar(results_frame, orient="horizontal",
                                  length=200, mode="determinate", maximum=100)
            pb.grid(row=i, column=2, sticky="w")

            self.model_labels[name]  = val
            self.progress_bars[name] = pb

        # —––––– LOAD MODELS –––––—
        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.le     = joblib.load(LABELENCODER_PATH)
            self.models = {n: joblib.load(p) for n,p in MODEL_FILES.items()}
        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load model files:\n{e}")
            master.quit()

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files","*.wav")])
        if not path:
            return
        
        self.file_label.config(text=os.path.basename(path))

        try:
            # MFCC and scaling
            y, sr       = librosa.load(path, sr=SAMPLE_RATE)
            mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
            feat        = np.mean(mfcc.T, axis=0).reshape(1, -1)
            feat_scaled = self.scaler.transform(feat)

            for name, model in self.models.items():
                label = "-"
                conf_text = ""
                conf = 0.0

                if name in ("FNN", "NN"):
                    # Keras FNN/CNN: predict returns softmax (1, n_classes)
                    proba    = model.predict(feat_scaled)           # (1, n_classes)
                    best_idx = np.argmax(proba, axis=1)             # (1,)
                    label    = self.le.inverse_transform(best_idx)[0]
                    conf     = proba[0, best_idx[0]] * 100
                    conf_text = f" ({conf:.1f}%)"

                else:
                    # KNN/SVM from scikit-learn
                    # 1) label
                    y_pred = model.predict(feat_scaled)             # (1,)
                    label  = self.le.inverse_transform(y_pred)[0]
                    # 2) confidence
                    proba  = model.predict_proba(feat_scaled)[0]    # (n_classes,)
                    best   = np.argmax(proba)
                    conf   = proba[best] * 100
                    conf_text = f" ({conf:.1f}%)"

                self.model_labels[name].config(text=f"{label}{conf_text}")
                self.progress_bars[name].config(value=conf)

        except Exception as e:
            messagebox.showerror("Processing Error", f"Failed to process file:\n{e}")



if __name__ == "__main__":
    root = tk.Tk()
    app  = GuitarClassifierApp(root)
    root.mainloop()
