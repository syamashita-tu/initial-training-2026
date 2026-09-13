"""
q01.py
基礎レベル1：録音ファイルを読み込み，10 msフレーム単位の音声区間ラベルを作成する．

入力音声：./syamashita-tu/vad/data/inputs/A1.wav
出力ラベル：./syamashita/vad/data/labels/A1.csv
出力図：./syamashita/vad/data/figures/A1_annotation.png

実行方法：
  python q01.py

注意：argparseは使わない．設定はファイル先頭の定数を書き換える．

入力例：
  0.35-1.20,1.55-2.80
"""

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

FS_EXPECTED = 16000
FRAME_SEC = 0.010
EPS = 1.0e-12

AUDIO_PATH = "./syamashita-tu/vad/data/inputs/A1.wav"
LABEL_DIR = "./syamashita-tu/vad/data/labels"
FIG_DIR = "./syamashita-tu/vad/data/figures"
LABEL_PATH = LABEL_DIR + "/A1.csv"
FIG_PATH = FIG_DIR + "/A1_annotation.png"


def read_mono_wav(file_path):
    """wavを読み込み，モノラルのfloat64配列に変換する．"""
    if not os.path.exists(file_path):
        raise FileNotFoundError("音声ファイルが見つかりません: " + file_path)

    x, fs = sf.read(file_path, dtype="float64")

    if x.ndim == 2:
        x = np.mean(x, axis=1)

    if fs != FS_EXPECTED:
        raise ValueError("サンプリング周波数が16000 Hzではありません: fs=" + str(fs))

    return x, fs


def frame_signal(x, frame_length, hop_length):
    """信号をフレームへ分割する．端の不足分は0で埋める．"""
    if len(x) == 0:
        raise ValueError("音声信号が空です．")

    n_frames = int(np.ceil(max(1, len(x) - frame_length) / hop_length)) + 1
    total_length = (n_frames - 1) * hop_length + frame_length
    pad_length = max(0, total_length - len(x))
    x_pad = np.pad(x, (0, pad_length))

    frames = np.zeros((n_frames, frame_length), dtype=np.float64)
    for m in range(n_frames):
        start = m * hop_length
        frames[m, :] = x_pad[start:start + frame_length]

    return frames


def short_time_log_energy(x, fs):
    """10 msフレームの短時間対数エネルギーを計算する．"""
    frame_length = int(round(FRAME_SEC * fs))
    hop_length = frame_length
    frames = frame_signal(x, frame_length, hop_length)
    window = np.hanning(frame_length)
    energy = 10.0 * np.log10(EPS + np.mean((frames * window) ** 2, axis=1))
    start_sec = np.arange(len(energy)) * FRAME_SEC
    end_sec = start_sec + FRAME_SEC

    return energy, start_sec, end_sec


def parse_intervals(text):
    """0.35-1.20,1.55-2.80 のような入力を音声区間のリストへ変換する．"""
    intervals = []
    text = text.strip()

    if text == "":
        return intervals

    blocks = text.split(",")
    for block in blocks:
        block = block.strip()
        if block == "":
            continue
        if "-" not in block:
            raise ValueError("区間は start-end の形式で入力してください: " + block)

        values = block.split("-")
        if len(values) != 2:
            raise ValueError("区間は start-end の形式で入力してください: " + block)

        start = float(values[0])
        end = float(values[1])

        if start < 0.0 or end <= start:
            raise ValueError("不正な区間です: " + block)

        intervals.append((start, end))

    return intervals


def make_frame_labels(start_sec, end_sec, intervals):
    """各10 msフレームに対して，音声なら1，非音声なら0を割り当てる．"""
    labels = np.zeros(len(start_sec), dtype=np.int64)

    for m in range(len(labels)):
        frame_center = 0.5 * (start_sec[m] + end_sec[m])
        for interval_start, interval_end in intervals:
            if interval_start <= frame_center < interval_end:
                labels[m] = 1
                break

    return labels


def save_label_file(file_path, start_sec, end_sec, labels):
    """ラベルをカンマ区切りのテキストとして保存する．"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("frame_index,start_sec,end_sec,label\n")
        for m in range(len(labels)):
            line = str(m) + ","
            line += format(start_sec[m], ".3f") + ","
            line += format(end_sec[m], ".3f") + ","
            line += str(int(labels[m])) + "\n"
            f.write(line)


def plot_annotation(x, fs, energy, start_sec, labels, save_path):
    """波形，短時間エネルギー，ラベルを確認するための図を保存する．"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    t = np.arange(len(x)) / fs

    plt.figure(figsize=(12, 7))

    plt.subplot(3, 1, 1)
    plt.plot(t, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform")

    plt.subplot(3, 1, 2)
    plt.plot(start_sec, energy)
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [dB]")
    plt.title("Short-time log energy")

    plt.subplot(3, 1, 3)
    plt.step(start_sec, labels, where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("Label")
    plt.ylim(-0.1, 1.1)
    plt.title("Speech label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    x, fs = read_mono_wav(AUDIO_PATH)
    energy, start_sec, end_sec = short_time_log_energy(x, fs)

    duration_sec = len(x) / fs
    print("音声ファイル:", AUDIO_PATH)
    print("サンプリング周波数:", fs, "Hz")
    print("音声長:", format(duration_sec, ".3f"), "s")
    print("音声区間を秒で入力してください．")
    print("例: 0.35-1.20,1.55-2.80")
    print("音声区間がない場合は空欄のままEnterを押してください．")

    text = input("speech intervals = ")
    intervals = parse_intervals(text)
    labels = make_frame_labels(start_sec, end_sec, intervals)

    save_label_file(LABEL_PATH, start_sec, end_sec, labels)
    plot_annotation(x, fs, energy, start_sec, labels, FIG_PATH)

    print("保存しました:", LABEL_PATH)
    print("確認用の図を保存しました:", FIG_PATH)
    print("音声フレーム数:", int(np.sum(labels)))
    print("全フレーム数:", len(labels))


if __name__ == "__main__":
    main()
