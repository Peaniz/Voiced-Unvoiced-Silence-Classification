#!/usr/bin/env python3
"""
MODULE TỐI ƯU NGƯỠNG SUV CLASSIFICATION
Tối ưu ngưỡng STE, ZCR, ST sử dụng ground truth từ file .lab
Chỉ sử dụng các hàm tự viết và built-in functions của Python/Numpy
"""

import os
import numpy as np
from audio_analyzer import AudioAnalyzer

def find_training_files(training_dir):
    """
    Tìm các file training có cả wav và lab
    
    Args:
        training_dir: Thư mục chứa file training
        
    Returns:
        List[Tuple]: Danh sách (wav_path, lab_path)
    """
    training_files = []
    
    if not os.path.exists(training_dir):
        return training_files
    
    # Tìm tất cả file .wav trong thư mục
    for filename in os.listdir(training_dir):
        if filename.endswith('.wav'):
            wav_path = os.path.join(training_dir, filename)
            lab_path = wav_path.replace('.wav', '.lab')
            
            # Kiểm tra file .lab tương ứng có tồn tại không
            if os.path.exists(lab_path):
                training_files.append((wav_path, lab_path))
    
    return training_files


def optimize_thresholds_with_ground_truth(training_files):
    """
    Tối ưu ngưỡng sử dụng ground truth từ file .lab
    
    Args:
        training_files: Danh sách file training
        
    Returns:
        Dict: Ngưỡng tối ưu
    """
    print("Đang tối ưu ngưỡng với ground truth...")
    
    # Khởi tạo analyzer
    analyzer = AudioAnalyzer(frame_length=0.025, frame_shift=0.010, sr=16000)
    
    # Thu thập tất cả features và labels
    all_ste_features = []
    all_zcr_features = []
    all_st_features = []
    all_true_labels = []
    
    print("Thu thập features và ground truth labels...")
    
    for wav_path, lab_path in training_files:
        print(f"  Xử lý: {os.path.basename(wav_path)}")
        
        try:
            # Load audio và ground truth
            audio, _ = analyzer.load_audio(wav_path)
            segments = analyzer.load_labels(lab_path)
            
            # Tính features
            ste = analyzer.compute_ste(audio)
            zcr = analyzer.compute_zcr(audio)
            st = analyzer.compute_spectrum_tilt(audio)
            
            # Tạo frame labels từ segments
            frame_labels = analyzer.get_frame_labels(segments, len(audio))
            
            # Đảm bảo chiều dài khớp nhau
            min_length = min(len(ste), len(zcr), len(st), len(frame_labels))
            
            # Thu thập features và labels
            all_ste_features.extend(ste[:min_length])
            all_zcr_features.extend(zcr[:min_length])
            all_st_features.extend(st[:min_length])
            all_true_labels.extend(frame_labels[:min_length])
            
        except Exception as e:
            print(f"    Lỗi xử lý {wav_path}: {e}")
            continue
    
    if len(all_true_labels) == 0:
        print("Không thu thập được dữ liệu training!")
        # Trả về ngưỡng mặc định
        return {
            'ste_threshold': -1.5,
            'zcr_threshold': 0.3,
            'st_threshold': 0.7,
            'frame_length': 0.025,
            'frame_shift': 0.010
        }
    
    # Chuyển sang numpy arrays
    ste_array = np.array(all_ste_features)
    zcr_array = np.array(all_zcr_features)
    st_array = np.array(all_st_features)
    labels_array = np.array(all_true_labels)
    
    print(f"Tổng cộng thu thập {len(labels_array)} frames với ground truth")
    
    # Tính phân bố class
    n_silence = np.sum(labels_array == 0)
    n_voiced = np.sum(labels_array == 1)
    n_unvoiced = np.sum(labels_array == 2)
    
    print(f"Phân bố class:")
    print(f"  Silence: {n_silence} frames ({n_silence/len(labels_array)*100:.1f}%)")
    print(f"  Voiced: {n_voiced} frames ({n_voiced/len(labels_array)*100:.1f}%)")
    print(f"  Unvoiced: {n_unvoiced} frames ({n_unvoiced/len(labels_array)*100:.1f}%)")
    
    # Tối ưu từng ngưỡng dựa trên ground truth
    optimal_thresholds = compute_optimal_thresholds(
        ste_array, zcr_array, st_array, labels_array
    )
    
    return optimal_thresholds


def compute_optimal_thresholds(ste_values, zcr_values, st_values, labels):
    """
    Tính ngưỡng tối ưu dựa trên ground truth
    
    Args:
        ste_values: STE features
        zcr_values: ZCR features
        st_values: ST features
        labels: Ground truth labels (0=silence, 1=voiced, 2=unvoiced)
        
    Returns:
        Dict: Ngưỡng tối ưu
    """
    print("Tính toán ngưỡng tối ưu...")
    
    # Tách features theo class
    silence_mask = (labels == 0)
    voiced_mask = (labels == 1)
    unvoiced_mask = (labels == 2)
    speech_mask = (labels == 1) | (labels == 2)  # voiced + unvoiced
    
    # 1. STE Threshold: Tách silence vs speech
    silence_ste = ste_values[silence_mask]
    speech_ste = ste_values[speech_mask]
    
    if len(silence_ste) > 0 and len(speech_ste) > 0:
        # Tìm ngưỡng tối ưu: trung bình của 90th percentile silence và 10th percentile speech
        silence_90th = np.percentile(silence_ste, 90)
        speech_10th = np.percentile(speech_ste, 10)
        ste_threshold = (silence_90th + speech_10th) / 2
    else:
        ste_threshold = np.percentile(ste_values, 30)  # Fallback
    
    # 2. ZCR Threshold: Tách voiced vs unvoiced
    voiced_zcr = zcr_values[voiced_mask]
    unvoiced_zcr = zcr_values[unvoiced_mask]
    
    if len(voiced_zcr) > 0 and len(unvoiced_zcr) > 0:
        # Voiced có ZCR thấp, unvoiced có ZCR cao
        voiced_90th = np.percentile(voiced_zcr, 90)
        unvoiced_10th = np.percentile(unvoiced_zcr, 10)
        zcr_threshold = (voiced_90th + unvoiced_10th) / 2
    else:
        zcr_threshold = 0.3  # Fallback
    
    # 3. ST Threshold: Tách voiced vs unvoiced
    voiced_st = st_values[voiced_mask]
    unvoiced_st = st_values[unvoiced_mask]
    
    if len(voiced_st) > 0 and len(unvoiced_st) > 0:
        # Voiced có ST cao (tần số thấp), unvoiced có ST thấp (tần số cao)
        voiced_10th = np.percentile(voiced_st, 10)
        unvoiced_90th = np.percentile(unvoiced_st, 90)
        st_threshold = (voiced_10th + unvoiced_90th) / 2
        # Clamp vào khoảng hợp lý
        st_threshold = max(0.3, min(0.9, st_threshold))
    else:
        st_threshold = 0.7  # Từ bài báo SUVDA
    
    print(f"Ngưỡng tối ưu tìm được:")
    print(f"  STE Threshold (Speech/Silence): {ste_threshold:.6f}")
    print(f"  ZCR Threshold (Voiced/Unvoiced): {zcr_threshold:.6f}")
    print(f"  ST Threshold (Voiced/Unvoiced): {st_threshold:.6f}")
    
    return {
        'ste_threshold': ste_threshold,
        'zcr_threshold': zcr_threshold,
        'st_threshold': st_threshold,
        'frame_length': 0.025,
        'frame_shift': 0.010
    }


def save_optimization_results(thresholds, results_dir):
    """
    Lưu kết quả tối ưu ngưỡng
    
    Args:
        thresholds: Ngưỡng tối ưu
        results_dir: Thư mục lưu kết quả
    """
    report_file = os.path.join(results_dir, "threshold_optimization_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== BÁO CÁO TỐI ƯU NGƯỠNG SUV CLASSIFICATION ===\n\n")
        f.write("Phương pháp: Supervised Learning với Ground Truth từ file .lab\n")
        f.write("Thuật toán: SUVDA (Speech/Unvoiced/Voiced Detection Algorithm)\n\n")
        
        f.write("THAM SỐ KHUNG:\n")
        f.write(f"Frame Length: {thresholds['frame_length']*1000:.0f}ms\n")
        f.write(f"Frame Shift: {thresholds['frame_shift']*1000:.0f}ms\n\n")
        
        f.write("NGƯỠNG TỐI ƯU:\n")
        f.write(f"STE Threshold (Speech/Silence): {thresholds['ste_threshold']:.6f}\n")
        f.write(f"ZCR Threshold (Voiced/Unvoiced): {thresholds['zcr_threshold']:.6f}\n")
        f.write(f"ST Threshold (Voiced/Unvoiced): {thresholds['st_threshold']:.6f}\n\n")
        
        f.write("LOGIC PHÂN LOẠI SUVDA:\n")
        f.write(f"1. Silence: STE < {thresholds['ste_threshold']:.4f}\n")
        f.write(f"2. Voiced: STE >= {thresholds['ste_threshold']:.4f} AND ST > {thresholds['st_threshold']:.4f} AND ZCR < {thresholds['zcr_threshold']:.4f}\n")
        f.write(f"3. Unvoiced: STE >= {thresholds['ste_threshold']:.4f} AND (ST <= {thresholds['st_threshold']:.4f} OR ZCR >= {thresholds['zcr_threshold']:.4f})\n")
    
    print(f"Đã lưu báo cáo tối ưu: {report_file}")
