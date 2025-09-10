import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os
from itertools import product

# =========================
# 1) LDM encode/decode
# =========================
def ldm_encode(x, step=0.02, x0=0.0):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    y_hat = np.zeros(N, dtype=np.float64)
    bits = np.zeros(N, dtype=np.uint8)
    prev = x0
    for n in range(N):
        bit = 1 if x[n] >= prev else 0
        bits[n] = bit
        prev = prev + (step if bit == 1 else -step)
        prev = np.clip(prev, -1.0, 1.0)
        y_hat[n] = prev
    return bits, y_hat

# =========================
# 2) Band-pass filter
# =========================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(x, fs, low=300.0, high=3400.0, order=4):
    if high >= fs * 0.49:
        high = fs * 0.49
    b, a = butter_bandpass(low, high, fs, order=order)
    return lfilter(b, a, x)

# =========================
# 3) Frame utilities
# =========================
def frame_indices(n, hop, win):
    starts = np.arange(0, n - win + 1, hop, dtype=int)
    ends = starts + win
    return starts, ends

def bit_alternation_rate(bits_frame):
    flips = np.sum(bits_frame[1:] != bits_frame[:-1])
    return flips / len(bits_frame) if len(bits_frame)>0 else 0.0

def zero_crossing_rate(x):
    signs = np.sign(x)
    signs[signs==0] = 1
    zc = np.sum(np.abs(np.diff(signs))) / 2.0
    return zc / len(x) if len(x)>0 else 0.0

def short_time_log_energy(x, eps=1e-8):
    if len(x)==0:
        return -1e8
    e = np.sum(x*x)/len(x)
    return 10.0*np.log10(e+eps)

def extract_features(bits, y_dec_bp, fs, frame_ms=25, hop_ms=10):
    win = int(round(frame_ms*fs/1000.0))
    hop = int(round(hop_ms*fs/1000.0))
    starts, ends = frame_indices(len(bits), hop, win)

    feats = []
    for s,e in zip(starts, ends):
        bar = bit_alternation_rate(bits[s:e])
        zcr = zero_crossing_rate(y_dec_bp[s:e])
        eng = short_time_log_energy(y_dec_bp[s:e])
        feats.append((bar, zcr, eng))

    feats = np.array(feats, dtype=np.float64)
    times = starts / fs
    return feats, times, win, hop

# =========================
# 4) Frame classification
# =========================
def classify_frames(feats, thr):
    bar, zcr, eng = feats[:,0], feats[:,1], feats[:,2]
    labels = np.empty(len(feats), dtype='<U1')

    SIL_ABS = -40
    sil_mask = (eng <= thr['eng_sil']) | (eng <= SIL_ABS)
    labels[sil_mask] = 'S'

    voiced_mask = (~sil_mask) & (bar <= thr['bar_lo']) & (zcr <= thr['zcr_lo'])
    labels[voiced_mask] = 'V'

    unvoiced_mask = (~sil_mask) & (~voiced_mask)
    labels[unvoiced_mask] = 'U'

    return labels

# =========================
# 5) Segment reconstruction
# =========================
def intervals_from_labels(times, labels, hop_ms):
    hop_sec = hop_ms/1000.0
    segs = []
    cur_label, cur_start = labels[0], times[0]
    for i in range(1,len(labels)):
        if labels[i] != cur_label:
            segs.append((cur_start, times[i], cur_label))
            cur_label, cur_start = labels[i], times[i]
    segs.append((cur_start, times[-1]+hop_sec, cur_label))
    return segs

# =========================
# 6) Load reference labels
# =========================
def load_label_file(label_path):
    segs = []
    with open(label_path,"r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts)==3:
                start,end,lab = parts
                if lab.lower() in ["sil","silence"]:
                    lab = "S"
                elif lab.lower() in ["u","uv","unvoiced"]:
                    lab = "U"
                elif lab.lower() in ["v","voiced"]:
                    lab = "V"
                segs.append((float(start), float(end), lab))
    return segs

# =========================
# 7) MAE between boundaries
# =========================
def boundary_mae(segs_pred, segs_ref):
    b_pred = []
    for s,e,_ in segs_pred:
        b_pred.append(s)
        b_pred.append(e)
    b_ref = []
    for s,e,_ in segs_ref:
        b_ref.append(s)
        b_ref.append(e)
    b_pred = np.array(sorted(b_pred))
    b_ref = np.array(sorted(b_ref))
    if len(b_ref)==0:
        return np.nan, b_pred, b_ref
    dists = []
    for br in b_ref:
        idx = np.argmin(np.abs(b_pred - br))
        dists.append(abs(b_pred[idx]-br))
    return float(np.mean(dists)), b_pred, b_ref

# =========================
# 8) Visualization
# =========================
def label_to_num(lab):
    if lab == 'S': return 0
    if lab == 'U': return 1
    if lab == 'V': return 2
    return -1

def expand_segments(segs, hop_ms, fs, total_len):
    hop = int(round(hop_ms*fs/1000.0))
    n_frames = total_len // hop
    arr = np.zeros(n_frames, dtype=int)
    for (s,e,L) in segs:
        start = int(round(s*fs/hop))
        end = int(round(e*fs/hop))
        arr[start:end] = label_to_num(L)
    return arr

def plot_suving(x, fs, times, labels, feats, segs_pred, segs_ref=None, thr=None, hop_ms=10, save_png='suving_result.png'):
    plt.figure(figsize=(14,12))

    # Waveform
    ax1 = plt.subplot(5,1,1)
    t = np.arange(len(x))/fs
    ax1.plot(t, x, color='black')
    ax1.set_title('Waveform ')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')

    # Segmentation
    ax2 = plt.subplot(5,1,2)
    pred_arr = expand_segments(segs_pred, hop_ms, fs, len(x))
    t_frames = np.arange(len(pred_arr)) * (hop_ms/1000.0)
    ax2.plot(t_frames, pred_arr, label="Predicted", color='blue', linestyle='--')
    if segs_ref is not None:
        ref_arr = expand_segments(segs_ref, hop_ms, fs, len(x))
        ax2.plot(t_frames, ref_arr, label="Reference", color='red', linewidth=2)
    ax2.set_title('Segmentation Result (0=S, 1=U, 2=V)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Label ID')
    ax2.set_yticks([0,1,2])
    ax2.set_yticklabels(['S','U','V'])
    ax2.legend()

    # Feature plots
    bar, zcr, eng = feats[:,0], feats[:,1], feats[:,2]

    ax3 = plt.subplot(5,1,3)
    ax3.plot(times, bar, label='BAR')
    ax3.set_ylabel('BAR')
    ax3.set_title('Bit Alternation Rate')
    if thr is not None:
        ax3.axhline(thr.get('bar_lo', np.nan), linestyle='--', color='green', label='bar_lo')
        ax3.axhline(thr.get('bar_hi', np.nan), linestyle=':', color='green', label='bar_hi')
        ax3.legend()

    ax4 = plt.subplot(5,1,4)
    ax4.plot(times, zcr, label='ZCR')
    ax4.set_ylabel('ZCR')
    ax4.set_title('Zero Crossing Rate')
    if thr is not None:
        ax4.axhline(thr.get('zcr_lo', np.nan), linestyle='--', color='purple', label='zcr_lo')
        ax4.axhline(thr.get('zcr_hi', np.nan), linestyle=':', color='purple', label='zcr_hi')
        ax4.legend()

    ax5 = plt.subplot(5,1,5)
    ax5.plot(times, eng, label='Energy (dB)')
    ax5.set_ylabel('Energy (dB)')
    ax5.set_xlabel('Time (s)')
    ax5.set_title('Short-time Log Energy')
    if thr is not None:
        ax5.axhline(thr.get('eng_sil', np.nan), linestyle='--', color='orange', label='eng_sil (trained)')
        ax5.axhline(-40, linestyle=':', color='red', label='SIL_ABS=-40dB')
        ax5.legend()

    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    plt.show()

# =========================
# 9) Labels utility
# =========================
def labels_from_file(label_path, times, hop_ms):
    segs = load_label_file(label_path)
    arr = np.empty(len(times), dtype='<U1')
    idx = 0
    for s,e,L in segs:
        while idx < len(times) and times[idx] < s:
            arr[idx] = 'S'
            idx +=1
        while idx < len(times) and times[idx] <= e:
            arr[idx] = L
            idx +=1
    while idx < len(times):
        arr[idx] = 'S'
        idx +=1
    return arr

def frame_mae(pred_labels, ref_labels):
    diff = pred_labels != ref_labels
    return np.mean(diff)

# =========================
# 10) Trực quan hóa threshold
# =========================
def visualize_thresholds(features, thr):
    bar, zcr, eng = features[:,0], features[:,1], features[:,2]

    plt.figure(figsize=(14,4))

    plt.subplot(1,3,1)
    plt.hist(bar, bins=50, color='skyblue', alpha=0.7)
    plt.axvline(thr['bar_lo'], color='green', linestyle='--', label='bar_lo')
    plt.axvline(thr['bar_hi'], color='green', linestyle=':', label='bar_hi')
    plt.title('BAR distribution with thresholds'); plt.xlabel('BAR'); plt.ylabel('Count'); plt.legend()

    plt.subplot(1,3,2)
    plt.hist(zcr, bins=50, color='plum', alpha=0.7)
    plt.axvline(thr['zcr_lo'], color='purple', linestyle='--', label='zcr_lo')
    plt.axvline(thr['zcr_hi'], color='purple', linestyle=':', label='zcr_hi')
    plt.title('ZCR distribution with thresholds'); plt.xlabel('ZCR'); plt.ylabel('Count'); plt.legend()

    plt.subplot(1,3,3)
    plt.hist(eng, bins=50, color='orange', alpha=0.7)
    plt.axvline(thr['eng_sil'], color='red', linestyle='--', label='eng_sil')
    plt.title('Energy distribution with thresholds'); plt.xlabel('Energy (dB)'); plt.ylabel('Count'); plt.legend()

    plt.tight_layout()
    plt.show()

# =========================
# 11) Grid search threshold
# =========================
def optimize_thresholds(files, frame_ms=25, hop_ms=10, ldm_step=0.02, target_fs=16000):
    all_feats = []
    all_labels = []

    for wav_path, lab_path in files:
        x, fs = librosa.load(wav_path, sr=target_fs, mono=True)
        x = x / (np.max(np.abs(x)) + 1e-12)
        bits, y_dec = ldm_encode(x, step=ldm_step)
        y_dec_bp = bandpass_filter(y_dec, fs)
        feats, times, win, hop = extract_features(bits, y_dec_bp, fs, frame_ms, hop_ms)
        labels = labels_from_file(lab_path, times, hop_ms)
        all_feats.append(feats)
        all_labels.append(labels)

    all_feats = np.vstack(all_feats)
    all_labels = np.concatenate(all_labels)

    bar_lo_range = np.linspace(0.1, 0.9, 5)
    zcr_lo_range = np.linspace(0.01, 0.3, 5)
    eng_sil_range = np.linspace(-50, -20, 5)

    best_mae = 1.0
    best_thr = None

    for bar_lo, zcr_lo, eng_sil in product(bar_lo_range, zcr_lo_range, eng_sil_range):
        thr = dict(bar_lo=bar_lo, zcr_lo=zcr_lo, eng_sil=eng_sil)
        thr['bar_hi'] = min(bar_lo+0.2,1.0)
        thr['zcr_hi'] = min(zcr_lo+0.2,1.0)
        labels_pred = classify_frames(all_feats, thr)
        mae = frame_mae(labels_pred, all_labels)
        if mae < best_mae:
            best_mae = mae
            best_thr = thr

    print("Training MAE:", best_mae)
    visualize_thresholds(all_feats, best_thr)
    return best_thr

# =========================
# 12) Test file
# =========================
def test_file(wav_path, lab_path, thr, frame_ms=25, hop_ms=10, ldm_step=0.02, target_fs=16000, save_png='suving_result.png'):
    x, fs = librosa.load(wav_path, sr=target_fs, mono=True)
    x = x / (np.max(np.abs(x)) + 1e-12)
    bits, y_dec = ldm_encode(x, step=ldm_step)
    y_dec_bp = bandpass_filter(y_dec, fs)
    feats, times, win, hop = extract_features(bits, y_dec_bp, fs, frame_ms, hop_ms)
    labels = classify_frames(feats, thr)
    segs_pred = intervals_from_labels(times, labels, hop_ms)
    segs_ref = load_label_file(lab_path)
    mae, b_pred, b_ref = boundary_mae(segs_pred, segs_ref)
    plot_suving(x, fs, times, labels, feats, segs_pred, segs_ref, thr, hop_ms, save_png)
    print(f"File: {wav_path}, MAE = {mae:.3f} sec")
    return mae

# =========================
# 13) Main
# =========================
if __name__=="__main__":
    files = [
        ("TinHieuHuanLuyen/phone_F1.wav", "TinHieuHuanLuyen/phone_F1.lab"),
        ("TinHieuHuanLuyen/phone_M1.wav", "TinHieuHuanLuyen/phone_M1.lab"),
        ("TinHieuHuanLuyen/studio_F1.wav", "TinHieuHuanLuyen/studio_F1.lab"),
        ("TinHieuHuanLuyen/studio_M1.wav", "TinHieuHuanLuyen/studio_M1.lab"),
    ]
    testfiles = [
        ("TinHieuKiemThu/phone_F2.wav", "TinHieuKiemThu/phone_F2.lab"),
        ("TinHieuKiemThu/phone_M2.wav", "TinHieuKiemThu/phone_M2.lab"),
        ("TinHieuKiemThu/studio_F2.wav", "TinHieuKiemThu/studio_F2.lab"),
        ("TinHieuKiemThu/studio_M2.wav", "TinHieuKiemThu/studio_M2.lab"),
    ]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Huấn luyện threshold
    thr_opt = optimize_thresholds(files)

    # Test từng file
    for i, (wav, lab) in enumerate(testfiles, 1):
        save_name = os.path.join(BASE_DIR, f"result_{i}.png")
        test_file(wav, lab, thr_opt, save_png=save_name)
