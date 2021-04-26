#!/usr/bin/env python
# coding: utf-8


import numpy as np
import librosa


def extract_dB_spectrogram(audio, n_fft, win_length, hop_length):
    
    amplitude_spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, win_length=win_length, hop_length=hop_length))    
    return librosa.amplitude_to_db(amplitude_spectrogram, np.max(amplitude_spectrogram))
