import numpy as np
import soundfile as sf

def load_audio_mono8(path, id): #soundfile で wav を読み込み，モノラル化
    return sf.read(path+id)
def frame_signal(x, fs): #10 ms 単位に分割

def short_time_energy(frames): #Hann 窓付き短時間対数エネルギー

def relative_energy(energy): #20 パーセンタイルを引く

def threshold_vad(): #スコアが $\theta$ 以上なら音声

def apply_hangover(): #音声直後の非音声を一定長だけ音声保持

def annotation_to_frame_labels(): #秒単位アノテーションからフレームラベル作成

def confusion_counts(): #混同行列

def macro_f1_from_counts(): #macro-F1

def search_best_threshold(): #train で最適な $\theta$ を探索

def search_best_relative_hangover(): #train で最適な， $L_h$ を探索

def main():
    test = [A1, A2, A3, A4, A5, A6]
    train = [A7, A8, A9]
    path = "./syamashita-tu/vad/data/inputs/"
