import streamlit as st
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import tempfile
import os
import sounddevice as sd

st.set_page_config(page_title="Фур'є-караоке", layout="centered")
st.title("🎤 Фур'є-караоке")


uploaded_file = st.file_uploader("Завантаж свій голосовий .wav файл", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    data, samplerate = sf.read(tmp_path)
    if len(data.shape) > 1:
        data = data[:, 0]

    st.audio(uploaded_file, format='audio/wav')

    fft_vals = fft(data)
    frequencies = fftfreq(len(data), 1 / samplerate)

    n_harmonics = st.slider("Кількість гармонік", min_value=512, max_value=32768, value=512, step=1024)

    def keep_n_harmonics(fft_vals, n):
        fft_copy = np.copy(fft_vals)
        sorted_indices = np.argsort(np.abs(fft_copy))[:-n]
        fft_copy[sorted_indices] = 0
        return fft_copy

    filtered_fft = keep_n_harmonics(fft_vals, n_harmonics)
    reconstructed = np.real(ifft(filtered_fft))

    st.subheader("🔊 Відтворення відновленого звуку")
    st.write(f"Гармонік використано: {n_harmonics}")

    norm_audio = reconstructed / np.max(np.abs(reconstructed))

    recon_path = tmp_path.replace(".wav", f"_recon_{n_harmonics}.wav")
    sf.write(recon_path, norm_audio, samplerate)

    with open(recon_path, 'rb') as audio_file:
        st.audio(audio_file.read(), format='audio/wav')


    os.remove(tmp_path)
else:
    st.info("Завантаж .wav файл, щоб почати")
