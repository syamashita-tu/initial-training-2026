import os
import numpy as np
import soundfile as sf


DATA_DIR = "data"
LABEL_DIR = "labels"

TRAIN_IDS = ["A1", "A2", "A3", "A4", "A5", "A6"]
TEST_IDS = ["A7", "A8", "A9"]

TARGET_FS = 16000
FRAME_SEC = 0.010
EPS = 1.0e-12
THETA_STEP = 0.5
HANGOVER_LIST = [0, 3, 5, 10, 20]


def read_wav_mono(wav_path):
    x, fs = sf.read(wav_path, dtype="float32")

    if fs != TARGET_FS:
        raise ValueError(
            wav_path + " のサンプリング周波数が16000 Hzではありません: " + str(fs)
        )

    if x.ndim == 2:
        x = np.mean(x, axis=1)

    return fs, x


def frame_signal(x, fs):
    frame_length = int(round(FRAME_SEC * fs))
    num_frames = int(np.ceil(len(x) / frame_length))
    pad_length = num_frames * frame_length - len(x)

    if pad_length > 0:
        x = np.pad(x, (0, pad_length))

    frames = x.reshape(num_frames, frame_length)

    return frames, frame_length


def compute_log_energy(frames):
    frame_length = frames.shape[1]
    window = np.hanning(frame_length)

    xw = frames * window[None, :]
    power = np.mean(xw ** 2, axis=1)
    energy_db = 10.0 * np.log10(EPS + power)

    return energy_db


def load_segment_csv(csv_path):
    try:
        segments = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    except Exception:
        segments = np.loadtxt(csv_path, delimiter=",")

    segments = np.asarray(segments, dtype=np.float64)

    if segments.ndim == 1:
        segments = segments.reshape(1, -1)

    if segments.shape[1] < 2:
        raise ValueError("CSV は start_sec,end_sec の2列が必要: " + csv_path)

    return segments[:, :2]


def segments_to_frame_labels(segments, num_frames):
    labels = np.zeros(num_frames, dtype=np.int64)
    frame_centers = (np.arange(num_frames) + 0.5) * FRAME_SEC

    for start_sec, end_sec in segments:
        labels[(frame_centers >= start_sec) & (frame_centers < end_sec)] = 1

    return labels


def load_labels(split, utt_id, num_frames):
    csv_path = os.path.join(LABEL_DIR, split, utt_id + ".csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError("正解ラベルCSVが見つかりません: " + csv_path)

    segments = load_segment_csv(csv_path)
    labels = segments_to_frame_labels(segments, num_frames)

    return labels


def load_record(split, utt_id):
    wav_path = os.path.join(DATA_DIR, split, utt_id + ".wav")

    fs, x = read_wav_mono(wav_path)
    frames, frame_length = frame_signal(x, fs)

    energy = compute_log_energy(frames)

    noise_floor = np.percentile(energy, 20)
    relative_energy = energy - noise_floor

    label = load_labels(split, utt_id, len(energy))

    return {
        "utt_id": utt_id,
        "energy": energy,
        "relative_energy": relative_energy,
        "label": label,
    }


def load_records(split, utt_ids):
    records = []

    for utt_id in utt_ids:
        records.append(load_record(split, utt_id))

    return records


def apply_hangover(z0, hangover_length):
    z0 = np.asarray(z0, dtype=np.int64)
    z = z0.copy()

    if hangover_length <= 0:
        return z

    num_frames = len(z0)

    for m in range(num_frames - 1):
        if z0[m] == 1 and z0[m + 1] == 0:
            end = min(num_frames, m + hangover_length + 1)
            z[m + 1:end] = 1

    return z


def macro_f1_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    if 2 * tp + fp + fn == 0:
        f1_speech = 0.0
    else:
        f1_speech = 2 * tp / (2 * tp + fp + fn)

    if 2 * tn + fn + fp == 0:
        f1_non_speech = 0.0
    else:
        f1_non_speech = 2 * tn / (2 * tn + fn + fp)

    macro_f1 = 0.5 * (f1_speech + f1_non_speech)

    return {
        "macro_f1": macro_f1,
        "f1_speech": f1_speech,
        "f1_non_speech": f1_non_speech,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def get_score(record, mode):
    if mode == "energy":
        return record["energy"]

    if mode == "relative_energy":
        return record["relative_energy"]

    raise ValueError("未知の mode: " + mode)


def evaluate(records, mode, theta, hangover_length):
    y_true_all = []
    y_pred_all = []

    for record in records:
        score = get_score(record, mode)

        z0 = (score >= theta).astype(np.int64)
        z = apply_hangover(z0, hangover_length)

        y_true_all.append(record["label"])
        y_pred_all.append(z)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    return macro_f1_score(y_true_all, y_pred_all)


def make_theta_candidates(records, mode):
    scores = []

    for record in records:
        scores.append(get_score(record, mode))

    scores = np.concatenate(scores)

    theta_min = np.floor(np.min(scores) / THETA_STEP) * THETA_STEP
    theta_max = np.ceil(np.max(scores) / THETA_STEP) * THETA_STEP

    return np.arange(theta_min, theta_max + THETA_STEP, THETA_STEP)


def search_best_theta(records, mode):
    best = None

    for theta in make_theta_candidates(records, mode):
        result = evaluate(
            records=records,
            mode=mode,
            theta=theta,
            hangover_length=0,
        )

        if best is None or result["macro_f1"] > best["macro_f1"]:
            best = result.copy()
            best["mode"] = mode
            best["theta"] = float(theta)
            best["hangover_length"] = 0

    return best


def search_best_theta_and_hangover(records, mode):
    best = None

    for hangover_length in HANGOVER_LIST:
        for theta in make_theta_candidates(records, mode):
            result = evaluate(
                records=records,
                mode=mode,
                theta=theta,
                hangover_length=hangover_length,
            )

            if best is None:
                update = True
            elif result["macro_f1"] > best["macro_f1"]:
                update = True
            elif result["macro_f1"] == best["macro_f1"]:
                update = hangover_length < best["hangover_length"]
            else:
                update = False

            if update:
                best = result.copy()
                best["mode"] = mode
                best["theta"] = float(theta)
                best["hangover_length"] = int(hangover_length)

    return best


def print_result(title, result):
    print("")
    print(title)
    print("mode              :", result["mode"])
    print("theta [dB]        :", result["theta"])
    print("hangover L_h      :", result["hangover_length"])
    print("macro-F1          :", result["macro_f1"])
    print("F1 speech         :", result["f1_speech"])
    print("F1 non-speech     :", result["f1_non_speech"])
    print("TP, FP, FN, TN    :", result["tp"], result["fp"], result["fn"], result["tn"])


def main():
    train_records = load_records("train", TRAIN_IDS)

    result_energy = search_best_theta(
        records=train_records,
        mode="energy",
    )

    result_relative = search_best_theta(
        records=train_records,
        mode="relative_energy",
    )

    result_relative_hangover = search_best_theta_and_hangover(
        records=train_records,
        mode="relative_energy",
    )

    print_result("1. エネルギーしきい値のみ", result_energy)
    print_result("2. 相対エネルギーしきい値", result_relative)
    print_result("3. 相対エネルギー + hangover", result_relative_hangover)


if __name__ == "__main__":
    main()