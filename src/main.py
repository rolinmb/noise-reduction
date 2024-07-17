import librosa
import numpy as np
import matplotlib.pyplot as plt

AUDIO_IN = 'audio-in/sesh_07172024.wav'
AUDIO_OUT = 'audio-out/sesh_07172024_redux.wav'

if __name__ == '__main__':
    y, sr = librosa.load(AUDIO_IN, sr=None)
    S = librosa.stft(y)
    mask = np.abs(S) > np.median(np.abs(S))
    S_clean = S * mask
    y_clean = librosa.istft(S_clean)
    librosa.output.write_wav(AUDIO_OUT, y_clean, sr)
