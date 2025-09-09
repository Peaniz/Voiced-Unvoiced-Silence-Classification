import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import os

# =========================
# 1) Linear Delta Modulation (LDM)
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
# 4) Adaptive thresholds
# =========================
def adaptive_thresholds(feats):
    bar, zcr, eng = feats[:,0], feats[:,1], feats[:,2]
    return dict(
        bar_lo = np.percentile(bar, 40),
        bar_hi = np.percentile(bar, 60),
        zcr_lo = np.percentile(zcr, 40),
        zcr_hi = np.percentile(zcr, 70),
        eng_sil = np.percentile(eng, 30)

    )

# =========================
# 5) Frame classification
# =========================
def classify_frames(feats, thr):
    bar, zcr, eng = feats[:,0], feats[:,1], feats[:,2]
    labels = np.empty(len(feats), dtype='<U1')

    # Silence
    SIL_ABS = -40
    sil_mask = (eng <= thr['eng_sil']) | (eng <= SIL_ABS)
    labels[sil_mask] = 'S'

    # Voiced
    voiced_mask = (~sil_mask) & (bar <= thr['bar_lo']) & (zcr <= thr['zcr_lo'])
    labels[voiced_mask] = 'V'

    # Unvoiced
    unvoiced_mask = (~sil_mask) & (~voiced_mask)
    labels[unvoiced_mask] = 'U'

    return labels

# =========================
# 6) Segment reconstruction
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
# 7) Load reference labels
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
# 8) MAE between boundaries
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
# 9) Visualization (modified)
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

    # --- 1) Waveform only ---
    ax1 = plt.subplot(5,1,1)
    t = np.arange(len(x))/fs
    ax1.plot(t, x, color='black')
    ax1.set_title('Waveform ')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')

    # --- 2) Segmentation comparison ---
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

    # --- 3) Feature plots ---
    bar, zcr, eng = feats[:,0], feats[:,1], feats[:,2]

    ax3 = plt.subplot(5,1,3)
    ax3.plot(times, bar)
    ax3.set_ylabel('BAR')
    ax3.set_title('Bit Alternation Rate')
    if thr is not None:
        ax3.axhline(thr.get('bar_lo', np.nan), linestyle='--', label='bar_lo')
        ax3.axhline(thr.get('bar_hi', np.nan), linestyle=':', label='bar_hi')
        ax3.legend()

    ax4 = plt.subplot(5,1,4)
    ax4.plot(times, zcr)
    ax4.set_ylabel('ZCR')
    ax4.set_title('Zero Crossing Rate')
    if thr is not None:
        ax4.axhline(thr.get('zcr_lo', np.nan), linestyle='--', label='zcr_lo')
        ax4.axhline(thr.get('zcr_hi', np.nan), linestyle=':', label='zcr_hi')
        ax4.legend()

    ax5 = plt.subplot(5,1,5)
    ax5.plot(times, eng)
    ax5.set_ylabel('Energy (dB)')
    ax5.set_xlabel('Time (s)')
    ax5.set_title('Short-time Log Energy')
    if thr is not None:
        ax5.axhline(thr.get('eng_sil', np.nan), linestyle='--', label='eng_sil (adaptive)')
        ax5.axhline(-40, linestyle=':', label='SIL_ABS = -40 dB')
        ax5.legend()

    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    plt.show()

# =========================
# 10) Full pipeline
# =========================
def suving_pipeline(wav_path, label_path, frame_ms=25, hop_ms=10, ldm_step=0.02, target_fs=16000, save_png='suving_result.png'):
    x, fs = librosa.load(wav_path, sr=target_fs, mono=True)
    x = x / (np.max(np.abs(x))+1e-12)

    bits, y_dec = ldm_encode(x, step=ldm_step)
    y_dec_bp = bandpass_filter(y_dec, fs, 300.0, 3400.0)

    feats, times, win, hop = extract_features(bits, y_dec_bp, fs, frame_ms, hop_ms)
    thr = adaptive_thresholds(feats)
    labels = classify_frames(feats, thr)
    segs_pred = intervals_from_labels(times, labels, hop_ms)

    segs_ref = load_label_file(label_path)
    mae, b_pred, b_ref = boundary_mae(segs_pred, segs_ref)

    return x, fs, times, labels, feats, segs_pred, segs_ref, thr, mae, b_pred, b_ref

# =========================
# 11) Example run
# =========================
if __name__=="__main__":
    files = [
        ("TinHieuHuanLuyen/phone_F1.wav", "TinHieuHuanLuyen/phone_F1.lab"),
        ("TinHieuHuanLuyen/phone_M1.wav", "TinHieuHuanLuyen/phone_M1.lab"),
        ("TinHieuHuanLuyen/studio_F1.wav", "TinHieuHuanLuyen/studio_F1.lab"),
        ("TinHieuHuanLuyen/studio_M1.wav", "TinHieuHuanLuyen/studio_M1.lab"),
    ]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    for i, (wav, lab) in enumerate(files, 1):
        save_name = os.path.join(BASE_DIR, f"result_{i}.png")
        x, fs, times, labels, feats, segs_pred, segs_ref, thr, mae, b_pred, b_ref = suving_pipeline(
            wav, lab, save_png=save_name
        )
        
        # gọi thêm hàm plot để vẽ và lưu biểu đồ
        plot_suving(
            x, fs, times, labels, feats, segs_pred, segs_ref, thr,
            save_png=save_name
        )
        
        print(f"Result {i}: MAE = {mae:.3f} sec (saved {save_name})")
