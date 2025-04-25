import numpy as np
import librosa 
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os

# DIRECTORIES WITH RECORDINGS AND SPECTROGRAMS
source_root = 'data'
output_root = 'Spectograms'

HOP = 64
N_FFT = 4096

for root, dirs, files in os.walk(source_root):
    for file in files:
        if file.lower().endswith('.wav'):
            filepath = os.path.join(root, file)

            # LOADING AUDIO
            y, sr = librosa.load(filepath, sr=None)

            # STFT AND CONVERSION TO dB
            D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
            D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            # OUTPUT PATH
            rel_path = os.path.relpath(filepath, source_root)
            rel_folder = os.path.dirname(rel_path)
            output_folder = os.path.join(output_root, rel_folder)
            os.makedirs(output_folder, exist_ok=True)

            # OUTPUT FILENAME
            filename_no_ext = os.path.splitext(file)[0]
            output_path = os.path.join(output_folder, f"{filename_no_ext}.png")

            # GENERATING SPECTROGRAM
            plt.figure(figsize=(12,6), dpi=300)
            librosa.display.specshow(D_db, sr=sr, hop_length=HOP, x_axis='time', y_axis='hz')
            plt.ylim(16, 20000)
            plt.axis('off')
            plt.margins(0)

            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"Saved: {output_path}")








#plt.title('Spectogram')
#plt.colorbar(format='%+2.0f dB')
#plt.gca().set_position([0, 0, 1, 1])  # stretches the spectrogram to cover the entire area