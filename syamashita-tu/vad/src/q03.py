"""
q03.py
基礎レベル3：VAD手法の改良を実装する．

改良方針：
  1. 全帯域の相対エネルギー
  2. 音声帯域 300-3400 Hz の相対エネルギー
  3. ゼロ交差率
を特徴量として用いる．trainに対して，特徴量重み，しきい値theta，hangover長Lhを探索し，
macro-F1が最大となる設定を選ぶ．

入力音声：./syamashita/vad/data/inputs/A1.wav から A6.wav
入力ラベル：./syamashita/vad/data/labels/A1.csv から A6.csv
出力結果：./syamashita/vad/data/results/q03_improved_results.csv

実行方法：
  python q03.py

注意：argparseは使わない．設定はファイル先頭の定数を書き換える．
"""

import os
import numpy as np
import soundfile as sf

FS_EXPECTED = 16000
FRAME_SEC = 0.010
EPS = 1.0e-12

INPUT_DIR = "./syamashita/vad/data/inputs"
LABEL_DIR = "./syamashita/vad/data/labels"
RESULT_DIR = "./syamashita/vad/data/results"
RESULT_PATH = RESULT_DIR + "/q03_improved_results.csv"

TRAIN_NAMES = ["A1", "A2", "A3", "A4", "A5", "A6"]
HANGOVER_LIST = [0, 3, 5, 10, 20]
WEIGHT_LIST = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.7, 0.3, 0.0),
    (0.5, 0.5, 0.0),
    (0.3, 0.7, 0.0),
    (0.6, 0.3, 0.1),
    (0.4, 0.5, 0.1),
    (0.5, 0.3, 0.2),
]


def read_mono_wav(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("音声ファイルが見つかりません: " + file_path)

    x, fs = sf.read(file_path, dtype="float64")

    if x.ndim == 2:
        x = np.mean(x, axis=1)

    if fs != FS_EXPECTED:
        raise ValueError("サンプリング周波数が16000 Hzではありません: " + file_path)

    return x, fs


def frame_signal(x, frame_length, hop_length):
    n_frames = int(np.ceil(max(1, len(x) - frame_length) / hop_length)) + 1
    total_length = (n_frames - 1) * hop_length + frame_length
    pad_length = max(0, total_length - len(x))
    x_pad = np.pad(x, (0, pad_length))

    frames = np.zeros((n_frames, frame_length), dtype=np.float64)
    for m in range(n_frames):
        start = m * hop_length
        frames[m, :] = x_pad[start:start + frame_length]

    return frames


def log_energy(frames, window):
    return 10.0 * np.log10(EPS + np.mean((frames * window) ** 2, axis=1))


def band_log_energy(frames, fs, window, low_hz, high_hz):
    xw = frames * window
    spectrum = np.fft.rfft(xw, axis=1)
    power = np.abs(spectrum) ** 2
    freq = np.fft.rfftfreq(frames.shape[1], d=1.0 / fs)
    band = (freq >= low_hz) & (freq <= high_hz)

    if np.sum(band) == 0:
        raise ValueError("指定した帯域に対応するFFTビンがありません．")

    return 10.0 * np.log10(EPS + np.mean(power[:, band], axis=1))


def zero_crossing_rate(frames):
    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = signs[:, 1:] * signs[:, :-1] < 0
    return np.mean(crossings, axis=1)


def read_label_file(file_path, n_frames):
    if not os.path.exists(file_path):
        raise FileNotFoundError("ラベルファイルが見つかりません．先にq01.pyで作成してください: " + file_path)

    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) <= 1:
        raise ValueError("ラベルファイルが空です: " + file_path)

    header = lines[0].strip().split(",")
    if "label" not in header:
        raise ValueError("label列が見つかりません: " + file_path)

    label_index = header.index("label")

    for line in lines[1:]:
        line = line.strip()
        if line == "":
            continue
        values = line.split(",")
        labels.append(int(values[label_index]))

    labels = np.array(labels, dtype=np.int64)

    if len(labels) < n_frames:
        labels = np.pad(labels, (0, n_frames - len(labels)), constant_values=0)
    elif len(labels) > n_frames:
        labels = labels[:n_frames]

    return labels


def normalize_feature(values):
    low = np.percentile(values, 5)
    high = np.percentile(values, 95)

    if high <= low:
        return np.zeros_like(values)

    normalized = (values - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def extract_features(x, fs):
    frame_length = int(round(FRAME_SEC * fs))
    frames = frame_signal(x, frame_length, frame_length)
    window = np.hanning(frame_length)

    energy = log_energy(frames, window)
    band_energy = band_log_energy(frames, fs, window, 300.0, 3400.0)
    zcr = zero_crossing_rate(frames)

    rel_energy = energy - np.percentile(energy, 20)
    rel_band_energy = band_energy - np.percentile(band_energy, 20)

    return rel_energy, rel_band_energy, zcr


def load_train_data():
    datasets = []

    for name in TRAIN_NAMES:
        audio_path = INPUT_DIR + "/" + name + ".wav"
        label_path = LABEL_DIR + "/" + name + ".csv"

        if not os.path.exists(audio_path) or not os.path.exists(label_path):
            continue

        x, fs = read_mono_wav(audio_path)
        rel_energy, rel_band_energy, zcr = extract_features(x, fs)
        label = read_label_file(label_path, len(rel_energy))

        datasets.append({
            "name": name,
            "rel_energy": rel_energy,
            "rel_band_energy": rel_band_energy,
            "zcr": zcr,
            "label": label
        })

    if len(datasets) == 0:
        raise RuntimeError("A1からA6の音声とラベルが見つかりません．q01.pyでA1.csvを作成してください．")

    return datasets


def apply_hangover(initial_pred, hangover_length):
    pred = initial_pred.copy()

    if hangover_length <= 0:
        return pred

    for m in range(len(initial_pred) - 1):
        if initial_pred[m] == 1 and initial_pred[m + 1] == 0:
            end = min(len(pred), m + 1 + hangover_length)
            pred[m + 1:end] = 1

    return pred


def f1_for_class(y_true, y_pred, target):
    tp = int(np.sum((y_true == target) & (y_pred == target)))
    fp = int(np.sum((y_true != target) & (y_pred == target)))
    fn = int(np.sum((y_true == target) & (y_pred != target)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def macro_f1(y_true, y_pred):
    f1_speech = f1_for_class(y_true, y_pred, 1)
    f1_non_speech = f1_for_class(y_true, y_pred, 0)
    return 0.5 * (f1_speech + f1_non_speech)


def make_score(data, weight_energy, weight_band, weight_zcr):
    e = normalize_feature(data["rel_energy"])
    b = normalize_feature(data["rel_band_energy"])
    z = normalize_feature(data["zcr"])

    # 音声はエネルギーが大きく，ゼロ交差率が極端に大きくなりにくいと仮定する．
    score = weight_energy * e + weight_band * b - weight_zcr * z
    return score


def make_thresholds(score):
    min_value = np.floor(np.min(score) * 20.0) / 20.0 - 0.05
    max_value = np.ceil(np.max(score) * 20.0) / 20.0 + 0.05
    return np.arange(min_value, max_value + 0.0001, 0.05)


def evaluate_setting(datasets, weights, theta, hangover_length):
    weight_energy, weight_band, weight_zcr = weights
    y_true_all = []
    y_pred_all = []

    for data in datasets:
        score = make_score(data, weight_energy, weight_band, weight_zcr)
        initial_pred = (score >= theta).astype(np.int64)
        pred = apply_hangover(initial_pred, hangover_length)

        y_true_all.append(data["label"])
        y_pred_all.append(pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    return macro_f1(y_true_all, y_pred_all)


def search_best(datasets):
    best = None

    for weights in WEIGHT_LIST:
        all_scores = []
        for data in datasets:
            all_scores.append(make_score(data, weights[0], weights[1], weights[2]))
        all_scores = np.concatenate(all_scores)
        theta_list = make_thresholds(all_scores)

        for theta in theta_list:
            for hangover_length in HANGOVER_LIST:
                score = evaluate_setting(datasets, weights, theta, hangover_length)
                current = {
                    "weight_energy": float(weights[0]),
                    "weight_band": float(weights[1]),
                    "weight_zcr": float(weights[2]),
                    "theta": float(theta),
                    "hangover": int(hangover_length),
                    "macro_f1": float(score)
                }

                if best is None:
                    best = current
                    continue

                better_score = current["macro_f1"] > best["macro_f1"]
                same_score_shorter_hangover = current["macro_f1"] == best["macro_f1"] and current["hangover"] < best["hangover"]

                if better_score or same_score_shorter_hangover:
                    best = current

    return best


def save_result(result):
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("method,weight_energy,weight_band,weight_zcr,theta,hangover,macro_f1\n")
        line = "improved_band_energy_zcr,"
        line += format(result["weight_energy"], ".3f") + ","
        line += format(result["weight_band"], ".3f") + ","
        line += format(result["weight_zcr"], ".3f") + ","
        line += format(result["theta"], ".3f") + ","
        line += str(result["hangover"]) + ","
        line += format(result["macro_f1"], ".6f") + "\n"
        f.write(line)


def main():
    datasets = load_train_data()
    result = search_best(datasets)
    save_result(result)

    print("使用したtrainデータ数:", len(datasets))
    print("改良手法: 全帯域相対エネルギー + 音声帯域相対エネルギー + ゼロ交差率")
    print("weight_energy =", result["weight_energy"])
    print("weight_band =", result["weight_band"])
    print("weight_zcr =", result["weight_zcr"])
    print("theta =", result["theta"])
    print("Lh =", result["hangover"])
    print("macro-F1 =", format(result["macro_f1"], ".6f"))
    print("保存しました:", RESULT_PATH)


if __name__ == "__main__":
    main()
