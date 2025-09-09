import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.stats import median_abs_deviation
import json, sys, os
import matplotlib.pyplot as plt

# ==== Config ====
frame_size = 80          # Will be adjusted based on actual sampling rate
hop = None               # Will be set to frame_size
window_size = 5
alpha_sd = 1.8
use_mad = True
bp = (80, 4000)

# ==== Utility ====
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    if fs <= 2 * highcut:
        highcut = fs // 2 - 100  # Adjust highcut if too close to nyquist
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, min(highcut/nyq, 0.95)], btype='band')
    return filtfilt(b, a, data)

def load_and_preprocess(path):
    fs, data = wavfile.read(path)
    print(f"  Sample rate: {fs} Hz, Duration: {len(data)/fs:.2f}s")
    
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    data -= np.mean(data)
    if np.max(np.abs(data)) > 0:
        data /= np.max(np.abs(data))
    
    # Apply bandpass filter if specified
    if bp and fs > 2 * bp[1]:
        data = butter_bandpass_filter(data, bp[0], bp[1], fs)
    
    return fs, data

def load_lab_file(lab_path):
    """Load ground truth labels from .lab file"""
    labels = []
    f0_mean = None
    f0_std = None
    print(f"  Loading labels from: {os.path.basename(lab_path)}")
    
    try:
        with open(lab_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  Error: Cannot find file {lab_path}")
        return [], None, None

    if len(lines) < 2:
        print(f"  Warning: Lab file has insufficient data")
        return [], None, None

    # Read segment lines (all except last 2)
    for line in lines[:-2]:
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.split('\t')  # Use tab separator
            if len(parts) >= 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                label = parts[2].strip()
                # Map label names to standardized format
                if label.lower() in ['silence', 'sil']:
                    label = 'sil'
                elif label.lower() in ['voiced', 'v']:
                    label = 'v'
                elif label.lower() in ['unvoiced', 'uv']:
                    label = 'uv'
                labels.append((start_time, end_time, label))
            else:
                print(f"  Format error in line (skipped): {line}")
        except (ValueError, IndexError) as e:
            print(f"  Error parsing line '{line}': {e}. Skipping.")

    # Read last two lines for F0 statistics
    if len(lines) >= 2:
        try:
            # F0mean line
            mean_line = lines[-2].strip()
            mean_parts = mean_line.split('\t')
            if len(mean_parts) >= 2 and mean_parts[0].lower().startswith("f0mean"):
                f0_mean = float(mean_parts[1])
            else:
                print(f"  F0mean format error (skipped): {mean_line}")

            # F0std line
            std_line = lines[-1].strip()
            std_parts = std_line.split('\t')
            if len(std_parts) >= 2 and std_parts[0].lower().startswith("f0std"):
                f0_std = float(std_parts[1])
            else:
                print(f"  F0std format error (skipped): {std_line}")

        except (ValueError, IndexError) as e:
            print(f"  Error parsing F0 statistics: {e}")
            f0_mean = None
            f0_std = None
        
    print(f"  ✓ Labels loaded: {len(labels)}, F0mean: {f0_mean}, F0std: {f0_std}")
    return labels, f0_mean, f0_std

def framing(data, frame_size, hop, fs):
    """Create frames and return frame times"""
    if hop is None: 
        hop = frame_size
    
    frames = []
    frame_times = []
    
    for i in range(0, len(data)-frame_size+1, hop):
        frames.append(data[i:i+frame_size])
        # Time at center of frame
        frame_times.append((i + frame_size/2) / fs)
    
    return np.array(frames), np.array(frame_times)

def get_frame_labels(frame_times, lab_segments):
    """Assign labels to frames based on lab file"""
    labels = []
    
    for frame_time in frame_times:
        # Find which segment this frame belongs to
        label = 'unknown'
        for start_time, end_time, seg_label in lab_segments:
            if start_time <= frame_time < end_time:
                label = seg_label
                break
        labels.append(label)
    
    return labels

def rms_energy(frame): 
    return np.sqrt(np.mean(frame**2))

def zcr(frame): 
    return np.sum(np.abs(np.diff(np.sign(frame))))/(2*len(frame))

def symbolic_transform(frame, window_size=5, alpha_sd=1.8, use_mad=True):
    """Symbolic dynamics transformation - matching btap1.py implementation"""
    if len(frame) < window_size:
        return []
    
    # Standardize the frame first (matching btap1.py)
    if np.std(frame) > 0:
        frame = (frame - np.mean(frame)) / np.std(frame)
    
    symbols = []
    for i in range(len(frame) - window_size + 1):
        window = frame[i:i + window_size]
        sd = np.std(window)
        if sd == 0:
            sd = 1e-10
        
        threshold = 1.0 * sd  # Matching btap1.py threshold calculation
        symbol_value = 0
        
        for j in range(window_size - 1):
            diff = abs(window[j] - window[j+1])
            if diff <= threshold:
                symbol_value += 1
        
        symbols.append(symbol_value)
    
    return symbols

def symbol_hist(symbols):
    if not symbols:
        return np.zeros(5)
    hist = np.zeros(5)
    for s in symbols:
        if 0 <= s <= 4: 
            hist[int(s)] += 1
    return hist / (np.sum(hist) or 1)

def entropy(hist):
    probs = hist[hist > 0]
    return -np.sum(probs * np.log2(probs)) if len(probs) else 0.0

def extract_features_with_labels(wav_path, lab_path):
    """Extract features and corresponding ground truth labels"""
    print(f"\nProcessing: {os.path.basename(wav_path)}")
    
    # Load audio
    fs, data = load_and_preprocess(wav_path)
    
    # Use fixed frame size of 80 to match btap1.py
    actual_frame_size = frame_size
    
    # Load ground truth
    lab_segments, f0_mean, f0_std = load_lab_file(lab_path)
    if not lab_segments:
        print(f"  Warning: No valid segments found in {lab_path}")
        return np.array([]), [], fs
        
    print(f"  Ground truth segments: {len(lab_segments)}")
    if f0_mean is not None and f0_std is not None:
        print(f"  F0: {f0_mean:.1f} ± {f0_std:.1f} Hz")
    
    # Create frames
    frames, frame_times = framing(data, actual_frame_size, actual_frame_size, fs)
    frame_labels = get_frame_labels(frame_times, lab_segments)
    
    print(f"  Total frames: {len(frames)}")
    
    # Count frames per class
    unique_labels, counts = np.unique(frame_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label != 'unknown':
            print(f"    {label}: {count} frames")
    
    # Extract features
    features = []
    labels = []
    
    for i, (frame, label) in enumerate(zip(frames, frame_labels)):
        if label == 'unknown':  # Skip frames without clear labels
            continue
            
        # Basic features
        e = rms_energy(frame)
        z = zcr(frame)
        
        # Symbolic dynamics features (matching btap1.py)
        syms = symbolic_transform(frame, window_size, alpha_sd, use_mad)
        hist = symbol_hist(syms)
        ent = entropy(hist)
        prop0 = np.mean(np.array(syms)==0) if len(syms) else 0
        
        # Feature vector: [energy, zcr, entropy, prop0, hist[0], hist[1], hist[2], hist[3], hist[4]]
        feat_vector = [e, z, ent, prop0] + hist.tolist()
        
        features.append(feat_vector)
        labels.append(label)
    
    print(f"  Valid feature vectors: {len(features)}")
    return np.array(features), labels, fs

def train_supervised_thresholds(wav_files, lab_files):
    """Train thresholds using supervised learning with ground truth labels"""
    
    if len(wav_files) != len(lab_files):
        raise ValueError("Number of wav files must match number of lab files")
    
    all_features = []
    all_labels = []
    
    # Collect all data
    print("="*50)
    print("COLLECTING TRAINING DATA")
    print("="*50)
    
    for wav_file, lab_file in zip(wav_files, lab_files):
        if os.path.exists(wav_file) and os.path.exists(lab_file):
            features, labels, fs = extract_features_with_labels(wav_file, lab_file)
            if len(features) > 0:
                all_features.append(features)
                all_labels.extend(labels)
            else:
                print(f"  Warning: No features extracted from {wav_file}")
        else:
            print(f"  Warning: Missing files - WAV: {os.path.exists(wav_file)}, LAB: {os.path.exists(lab_file)}")
            
    if not all_features:
        raise RuntimeError("No valid training data found.")
    
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    print(f"\nTotal training samples: {len(all_features)}")
    
    # Separate by class
    sil_features = all_features[all_labels == 'sil']
    v_features = all_features[all_labels == 'v'] 
    uv_features = all_features[all_labels == 'uv']
    
    print(f"Silence frames: {len(sil_features)}")
    print(f"Voiced frames: {len(v_features)}")
    print(f"Unvoiced frames: {len(uv_features)}")
    
    if len(sil_features) == 0:
        print("Warning: No silence samples found")
        silence_energy_threshold = 0.01
        silence_prop0_threshold = 0.85
    else:
        sil_energy = sil_features[:, 0]
        sil_prop0 = sil_features[:, 3]
        silence_energy_threshold = np.percentile(sil_energy, 95)
        silence_prop0_threshold = np.percentile(sil_prop0, 25)
    
    if len(v_features) == 0 or len(uv_features) == 0:
        print("Warning: Insufficient voiced/unvoiced samples")
        unvoiced_zcr_threshold = 0.2
        entropy_threshold = 0.73
    else:
        v_zcr = v_features[:, 1]
        uv_zcr = uv_features[:, 1]
        v_entropy = v_features[:, 2]
        uv_entropy = uv_features[:, 2]
        
        # Calculate optimal thresholds
        unvoiced_zcr_threshold = (np.median(v_zcr) + np.median(uv_zcr)) / 2
        entropy_threshold = (np.mean(v_entropy) + np.mean(uv_entropy)) / 2
    
    thresholds = {
        "silence_energy": float(silence_energy_threshold),
        "silence_prop0": float(silence_prop0_threshold), 
        "unvoiced_zcr": float(unvoiced_zcr_threshold),
        "entropy_mid": float(entropy_threshold),
        "feature_stats": {
            "silence": {
                "energy_mean": float(np.mean(sil_features[:, 0])) if len(sil_features) > 0 else 0.0,
                "energy_std": float(np.std(sil_features[:, 0])) if len(sil_features) > 0 else 0.0,
                "prop0_mean": float(np.mean(sil_features[:, 3])) if len(sil_features) > 0 else 0.0,
                "prop0_std": float(np.std(sil_features[:, 3])) if len(sil_features) > 0 else 0.0
            },
            "voiced": {
                "zcr_mean": float(np.mean(v_features[:, 1])) if len(v_features) > 0 else 0.0,
                "zcr_std": float(np.std(v_features[:, 1])) if len(v_features) > 0 else 0.0,
                "entropy_mean": float(np.mean(v_features[:, 2])) if len(v_features) > 0 else 0.0,
                "entropy_std": float(np.std(v_features[:, 2])) if len(v_features) > 0 else 0.0
            },
            "unvoiced": {
                "zcr_mean": float(np.mean(uv_features[:, 1])) if len(uv_features) > 0 else 0.0,
                "zcr_std": float(np.std(uv_features[:, 1])) if len(uv_features) > 0 else 0.0,
                "entropy_mean": float(np.mean(uv_features[:, 2])) if len(uv_features) > 0 else 0.0,
                "entropy_std": float(np.std(uv_features[:, 2])) if len(uv_features) > 0 else 0.0
            }
        }
    }
    
    # Print summary
    print(f"\n" + "="*50)
    print("TRAINED THRESHOLDS SUMMARY")
    print("="*50)
    print(f"Silence Energy Threshold: {silence_energy_threshold:.6f}")
    print(f"Silence Prop0 Threshold: {silence_prop0_threshold:.3f}")
    print(f"Unvoiced ZCR Threshold: {unvoiced_zcr_threshold:.4f}")
    print(f"Entropy Mid Threshold: {entropy_threshold:.3f}")
    
    return thresholds

def visualize_training_data(wav_files, lab_files, save_path=None):
    """Visualize the training data distribution"""
    
    all_features = []
    all_labels = []
    
    for wav_file, lab_file in zip(wav_files, lab_files):
        if os.path.exists(wav_file) and os.path.exists(lab_file):
            features, labels, _ = extract_features_with_labels(wav_file, lab_file)
            if len(features) > 0:
                all_features.append(features)
                all_labels.extend(labels)
    
    if not all_features:
        print("No data available for visualization")
        return
        
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Data Feature Distributions', fontsize=14)
    
    colors = {'sil': 'lightblue', 'v': 'red', 'uv': 'orange'}
    
    # Energy vs ZCR
    ax1 = axes[0, 0]
    for label in ['sil', 'v', 'uv']:
        mask = all_labels == label
        if np.any(mask):
            ax1.scatter(all_features[mask, 0], all_features[mask, 1], 
                       c=colors[label], alpha=0.6, label=label, s=20)
    ax1.set_xlabel('RMS Energy')
    ax1.set_ylabel('ZCR')
    ax1.set_title('Energy vs ZCR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Entropy distribution
    ax2 = axes[0, 1]
    for label in ['sil', 'v', 'uv']:
        mask = all_labels == label
        if np.any(mask):
            ax2.hist(all_features[mask, 2], alpha=0.6, color=colors[label], 
                    label=label, bins=20)
    ax2.set_xlabel('Entropy')
    ax2.set_ylabel('Count')
    ax2.set_title('Entropy Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Prop0 distribution
    ax3 = axes[1, 0]
    for label in ['sil', 'v', 'uv']:
        mask = all_labels == label
        if np.any(mask):
            ax3.hist(all_features[mask, 3], alpha=0.6, color=colors[label], 
                    label=label, bins=20)
    ax3.set_xlabel('Proportion of Symbol 0')
    ax3.set_ylabel('Count')
    ax3.set_title('Symbol 0 Proportion Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 2D feature space (Entropy vs Prop0)
    ax4 = axes[1, 1]
    for label in ['sil', 'v', 'uv']:
        mask = all_labels == label
        if np.any(mask):
            ax4.scatter(all_features[mask, 2], all_features[mask, 3], 
                       c=colors[label], alpha=0.6, label=label, s=20)
    ax4.set_xlabel('Entropy')
    ax4.set_ylabel('Proportion of Symbol 0')
    ax4.set_title('Entropy vs Prop0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training data visualization saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print("Usage: python train.py wav1 lab1 wav2 lab2 ...")
        print("Example: python train.py phone_F1.wav phone_F1.lab studio_M1.wav studio_M1.lab")
        sys.exit(1)
    
    # Parse arguments
    args = sys.argv[1:]
    wav_files = args[::2]  # Even indices
    lab_files = args[1::2]  # Odd indices
    
    print("="*60)
    print("SUPERVISED THRESHOLD TRAINING FOR SYMBOLIC DYNAMICS")
    print("="*60)
    print(f"WAV files: {[os.path.basename(f) for f in wav_files]}")
    print(f"LAB files: {[os.path.basename(f) for f in lab_files]}")
    
    # Check if files exist
    missing_files = []
    for wav_file, lab_file in zip(wav_files, lab_files):
        if not os.path.exists(wav_file):
            missing_files.append(f"WAV: {wav_file}")
        if not os.path.exists(lab_file):
            missing_files.append(f"LAB: {lab_file}")
    
    if missing_files:
        print("Error: Missing files:")
        for missing in missing_files:
            print(f"  {missing}")
        sys.exit(1)
    
    try:
        # Visualize training data
        print("\nGenerating training data visualization...")
        visualize_training_data(wav_files, lab_files, "training_data_visualization.png")
        
        # Train thresholds
        print("\nTraining supervised thresholds...")
        thresholds = train_supervised_thresholds(wav_files, lab_files)
        
        # Save thresholds
        output_file = "trained_thresholds.json"
        with open(output_file, "w") as f:
            json.dump(thresholds, f, indent=2)
        
        print(f"\n✓ Supervised thresholds saved to: {output_file}")
        print("\nYou can now use these thresholds with btap1.py:")
        print(f"python btap1.py your_audio.wav")
        print("(The trained thresholds will be loaded automatically)")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)