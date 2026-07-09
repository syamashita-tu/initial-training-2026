import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

fs = 16000
duration = 5
channels = 1

print("録音開始")
audio = sd.rec(
    int(duration * fs),
    samplerate=fs,
    channels=channels,
    dtype="float32",
)
sd.wait()
print("録音終了")

save_path = "./syamashita-tu/vad/data/inputs/A9.wav"
sf.write(save_path, audio, fs)

print("保存先:", save_path)