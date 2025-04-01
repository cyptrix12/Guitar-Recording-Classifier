import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

class FileCopierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Guitar Recording Organizer")

        # --- FOLDERS ---
        self.input_folder = ""
        self.output_folder = ""

        # --- MODE ---
        self.operation_mode = tk.StringVar(value="copy")

        # --- GUI Layout ---
        self.label = ttk.Label(root, text="Choose folders to start.")
        self.label.pack(pady=10)

        # --- INPUT FOLDER ---
        self.input_path_label = ttk.Label(root, text="(no input folder selected)", foreground="gray")
        self.input_path_label.pack()
        ttk.Button(root, text="Select Input Folder", command=self.select_input_folder).pack(pady=5)

        # --- OUTPUT FOLDER ---
        self.output_path_label = ttk.Label(root, text="(no output folder selected)", foreground="gray")
        self.output_path_label.pack()
        ttk.Button(root, text="Select Output Folder", command=self.select_output_folder).pack(pady=5)

        # --- COPY / MOVE OPTION ---
        mode_frame = ttk.LabelFrame(root, text="Choose Operation")
        mode_frame.pack(pady=5)
        ttk.Radiobutton(mode_frame, text="Copy files", variable=self.operation_mode, value="copy").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Move files", variable=self.operation_mode, value="move").pack(side=tk.LEFT, padx=10)

        # --- START BUTTON ---
        self.start_button = ttk.Button(root, text="Start", command=self.start_copying)
        self.start_button.pack(pady=10)

        # --- PROGRESS ---
        self.progress = ttk.Progressbar(root, length=400, mode='determinate')
        self.progress.pack(pady=5)

        # --- LOG WINDOW ---
        self.text_box = tk.Text(root, height=15, width=70)
        self.text_box.pack(pady=10)
        scrollbar = ttk.Scrollbar(root, command=self.text_box.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_box.configure(yscrollcommand=scrollbar.set)

    def log(self, msg):
        self.text_box.insert(tk.END, msg + "\n")
        self.text_box.see(tk.END)
        self.root.update_idletasks()

    def select_input_folder(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = filedialog.askdirectory(title="Select INPUT folder", initialdir=script_dir)
        if folder:
            self.input_folder = folder
            self.input_path_label.config(text=folder, foreground="black")
            self.log(f"Selected input folder: {folder}")

    def select_output_folder(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder = filedialog.askdirectory(title="Select OUTPUT folder", initialdir=script_dir)
        if folder:
            self.output_folder = folder
            self.output_path_label.config(text=folder, foreground="black")
            self.log(f"Selected output folder: {folder}")

    def start_copying(self):
        if not self.input_folder or not self.output_folder:
            messagebox.showerror("Error", "Please select both input and output folders.")
            return
        
        input_path = os.path.abspath(self.input_folder)
        output_path = os.path.abspath(self.output_folder)

        if input_path == output_path:
            messagebox.showerror("Error", "Input and output folders must be different!")
            return

        if input_path.startswith(output_path + os.sep) or output_path.startswith(input_path + os.sep):
            messagebox.showerror("Error", "Input and output folders must not be nested inside each other!")
            return

        self.start_button.config(state=tk.DISABLED)
        threading.Thread(target=self.copy_or_move_files).start()

    def copy_or_move_files(self):
        wav_files = []
        for root_dir, _, files in os.walk(self.input_folder):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append((file, root_dir))

        total = len(wav_files)
        if total == 0:
            self.log("No .wav files found.")
            self.start_button.config(state=tk.NORMAL)
            return

        self.progress.config(maximum=total)

        for idx, (file, source_dir) in enumerate(wav_files, 1):
            guitar_name = file.split('_')[0]
            target_dir = os.path.join(self.output_folder, guitar_name)
            os.makedirs(target_dir, exist_ok=True)

            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(target_dir, file)

            msg = f"{'Copying' if self.operation_mode.get() == 'copy' else 'Moving'} '{file}' from '{source_dir}' to '{target_dir}'"
            self.label.config(text=msg)
            self.log(msg)

            try:
                if self.operation_mode.get() == "copy":
                    shutil.copy2(src_path, dst_path)
                else:
                    shutil.move(src_path, dst_path)
            except Exception as e:
                self.log(f"Error processing {file}: {e}")

            self.progress['value'] = idx
            self.root.update_idletasks()

        self.label.config(text="✅ Operation complete.")
        self.log("✅ All files processed.")
        self.start_button.config(state=tk.NORMAL)

# --- RUN APP ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FileCopierApp(root)
    root.mainloop()
