"""
q02.py
基礎レベル2：短時間エネルギーに基づくベースラインVADを実装し，
trainに対するmacro-F1でしきい値thetaとhangover長Lhを決定する．

入力音声：./syamashita/vad/data/inputs/A1.wav から A6.wav
入力ラベル：./syamashita/vad/data/labels/A1.csv から A6.csv
出力結果：./syamashita/vad/data/results/q02_baseline_results.csv

実行方法：
  python q02.py

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
RESULT_PATH = RESULT_DIR + "/q02_baseline_results.csv"

TRAIN_NAMES = ["A1", "A2", "A3", "A4", "A5", "A6"]
HANGOVER_LIST = [0, 3, 5, 10, 20]


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


def short_time_log_energy(x, fs):
    frame_length = int(round(FRAME_SEC * fs))
    frames = frame_signal(x, frame_length, frame_length)
    window = np.hanning(frame_length)
    energy = 10.0 * np.log10(EPS + np.mean((frames * window) ** 2, axis=1))
    return energy


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


def load_train_data():
    datasets = []

    for name in TRAIN_NAMES:
        audio_path = INPUT_DIR + "/" + name + ".wav"
        label_path = LABEL_DIR + "/" + name + ".csv"

        if not os.path.exists(audio_path) or not os.path.exists(label_path):
            continue

        x, fs = read_mono_wav(audio_path)
        energy = short_time_log_energy(x, fs)
        label = read_label_file(label_path, len(energy))
        noise_floor = np.percentile(energy, 20)
        relative_energy = energy - noise_floor

        datasets.append({
            "name": name,
            "energy": energy,
            "relative_energy": relative_energy,
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


def make_thresholds(score):
    min_value = np.floor(np.min(score) * 2.0) / 2.0 - 1.0
    max_value = np.ceil(np.max(score) * 2.0) / 2.0 + 1.0
    return np.arange(min_value, max_value + 0.001, 0.5)


def evaluate_setting(datasets, score_name, theta, hangover_length):
    y_true_all = []
    y_pred_all = []

    for data in datasets:
        score = data[score_name]
        initial_pred = (score >= theta).astype(np.int64)
        pred = apply_hangover(initial_pred, hangover_length)

        y_true_all.append(data["label"])
        y_pred_all.append(pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    return macro_f1(y_true_all, y_pred_all)


def search_best(datasets, condition_name, score_name, hangover_candidates):
    all_scores = np.concatenate([data[score_name] for data in datasets])
    theta_list = make_thresholds(all_scores)

    best = None

    for theta in theta_list:
        for hangover_length in hangover_candidates:
            score = evaluate_setting(datasets, score_name, theta, hangover_length)
            current = {
                "condition": condition_name,
                "score_name": score_name,
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


def save_results(results):
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("condition,score_name,theta,hangover,macro_f1\n")
        for result in results:
            line = result["condition"] + ","
            line += result["score_name"] + ","
            line += format(result["theta"], ".3f") + ","
            line += str(result["hangover"]) + ","
            line += format(result["macro_f1"], ".6f") + "\n"
            f.write(line)


def main():
    datasets = load_train_data()

    results = []
    results.append(search_best(datasets, "energy_threshold_only", "energy", [0]))
    results.append(search_best(datasets, "relative_energy_threshold", "relative_energy", [0]))
    results.append(search_best(datasets, "relative_energy_plus_hangover", "relative_energy", HANGOVER_LIST))

    save_results(results)

    print("使用したtrainデータ数:", len(datasets))
    for result in results:
        print(result["condition"])
        print("  theta =", result["theta"])
        print("  Lh =", result["hangover"])
        print("  macro-F1 =", format(result["macro_f1"], ".6f"))

    print("保存しました:", RESULT_PATH)


if __name__ == "__main__":
    main()
