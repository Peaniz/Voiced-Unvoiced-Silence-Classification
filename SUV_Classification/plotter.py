"""
Module chứa các hàm vẽ biểu đồ cho SUV Classification
Tách riêng từ main.py để code gọn gàng và dễ bảo trì
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_comparison_with_ground_truth(audio, ste, zcr, st, predictions, smoothed_predictions,
                                    true_labels, filename, thresholds, figure_position, results_dir):
    """
    Vẽ kết quả so sánh với ground truth
    
    Args:
        audio: Tín hiệu âm thanh
        ste, zcr, st: Features (STE, ZCR, Spectrum Tilt)
        predictions: Dự đoán raw
        smoothed_predictions: Dự đoán smoothed
        true_labels: Ground truth từ file .lab
        filename: Tên file
        thresholds: Ngưỡng sử dụng
        figure_position: Vị trí figure (0-3)
        results_dir: Thư mục lưu kết quả
    """
    # Tạo figure với kích thước lớn
    fig = plt.figure(figsize=(15, 12))
    
    # Đặt vị trí cửa sổ ở 4 góc màn hình
    positions = [(50, 50), (800, 50), (50, 500), (800, 500)]
    if figure_position < len(positions):
        try:
            mngr = fig.canvas.manager
            if hasattr(mngr, 'window'):
                mngr.window.wm_geometry(f"+{positions[figure_position][0]}+{positions[figure_position][1]}")
        except:
            pass  # Ignore nếu không thể set window position
    
    # Tạo 7 subplots
    axes = fig.subplots(7, 1)
    
    # Tính thời gian cho audio và features
    sr = 16000
    hop_size = thresholds['frame_shift']
    time_audio = np.arange(len(audio)) / sr
    time_features = np.arange(len(ste)) * hop_size
    
    # Tạo title chính
    clean_name = filename.replace('.wav', '').replace('_', ' ').title()
    fig.suptitle(f'SUV Classification with Ground Truth - {clean_name}', 
                fontsize=14, fontweight='bold')
    
    # 1. Tín hiệu âm thanh gốc
    axes[0].plot(time_audio, audio, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_title('Tín hiệu âm thanh gốc', fontsize=10)
    axes[0].set_ylabel('Amplitude', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Short-time Energy (STE) với ngưỡng
    axes[1].plot(time_features, ste, 'g-', linewidth=1.5)
    axes[1].axhline(y=thresholds['ste_threshold'], color='red', linestyle='--', alpha=0.8, 
                   label=f"STE Threshold={thresholds['ste_threshold']:.3f}")
    axes[1].set_title('Short-time Energy (STE) - Phân biệt Speech vs Silence', fontsize=10)
    axes[1].set_ylabel('STE', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Zero Crossing Rate (ZCR) với ngưỡng
    axes[2].plot(time_features, zcr, 'm-', linewidth=1.5)
    axes[2].axhline(y=thresholds['zcr_threshold'], color='red', linestyle='--', alpha=0.8,
                   label=f"ZCR Threshold={thresholds['zcr_threshold']:.3f}")
    axes[2].set_title('Zero Crossing Rate (ZCR) - Phân biệt Voiced vs Unvoiced', fontsize=10)
    axes[2].set_ylabel('ZCR', fontsize=9)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # 4. Spectrum Tilt (ST) với ngưỡng
    if len(st) > 0:
        axes[3].plot(time_features[:len(st)], st, 'orange', linewidth=1.5)
        axes[3].axhline(y=thresholds['st_threshold'], color='red', linestyle='--', alpha=0.8,
                       label=f"ST Threshold={thresholds['st_threshold']:.3f}")
        axes[3].set_title('Spectrum Tilt (ST) - SUVDA Algorithm', fontsize=10)
        axes[3].set_ylabel('ST', fontsize=9)
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)
    
    # 5. Ground Truth Labels
    axes[4].plot(time_features[:len(true_labels)], true_labels, 'g-', 
                linewidth=2.5, label='Ground Truth')
    axes[4].set_title('Ground Truth Labels (từ file .lab)', fontsize=10, fontweight='bold')
    axes[4].set_ylabel('Label', fontsize=9)
    axes[4].set_ylim(-0.5, 2.5)
    axes[4].set_yticks([0, 1, 2])
    axes[4].set_yticklabels(['Silence', 'Voiced', 'Unvoiced'], fontsize=8)
    axes[4].legend(fontsize=8)
    axes[4].grid(True, alpha=0.3)
    
    # 6. Dự đoán Raw (trước khi làm mịn)
    axes[5].plot(time_features[:len(predictions)], predictions, 'b-', 
                linewidth=2, alpha=0.8, label='Raw Prediction')
    axes[5].set_title('Kết quả phân loại Raw (trước khi làm mịn)', fontsize=10)
    axes[5].set_ylabel('Label', fontsize=9)
    axes[5].set_ylim(-0.5, 2.5)
    axes[5].set_yticks([0, 1, 2])
    axes[5].set_yticklabels(['Silence', 'Voiced', 'Unvoiced'], fontsize=8)
    axes[5].legend(fontsize=8)
    axes[5].grid(True, alpha=0.3)
    
    # 7. So sánh Final Prediction vs Ground Truth
    min_len = min(len(true_labels), len(smoothed_predictions))
    axes[6].plot(time_features[:min_len], true_labels[:min_len], 'g-', 
                linewidth=2.5, alpha=0.8, label='Ground Truth')
    axes[6].plot(time_features[:min_len], smoothed_predictions[:min_len], 'r--', 
                linewidth=2, alpha=0.8, label='Final Prediction')
    
    # Tính và hiển thị accuracy
    accuracy = _compute_accuracy(true_labels[:min_len], smoothed_predictions[:min_len])
    axes[6].set_title(f'So sánh Final vs Ground Truth (Accuracy: {accuracy:.4f})', 
                     fontsize=10, fontweight='bold')
    axes[6].set_xlabel('Thời gian (s)', fontsize=9)
    axes[6].set_ylabel('Label', fontsize=9)
    axes[6].set_ylim(-0.5, 2.5)
    axes[6].set_yticks([0, 1, 2])
    axes[6].set_yticklabels(['Silence', 'Voiced', 'Unvoiced'], fontsize=8)
    axes[6].legend(fontsize=8)
    axes[6].grid(True, alpha=0.3)
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu figure
    save_path = os.path.join(results_dir, f"{filename.replace('.wav', '')}_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"    Đã lưu biểu đồ: {save_path}")


def plot_results_without_ground_truth(audio, ste, zcr, st, predictions, smoothed_predictions,
                                    filename, thresholds, figure_position, results_dir):
    """
    Vẽ kết quả phân loại không có ground truth
    
    Args:
        audio: Tín hiệu âm thanh
        ste, zcr, st: Features (STE, ZCR, Spectrum Tilt)
        predictions: Dự đoán raw
        smoothed_predictions: Dự đoán smoothed
        filename: Tên file
        thresholds: Ngưỡng sử dụng
        figure_position: Vị trí figure (0-3)
        results_dir: Thư mục lưu kết quả
    """
    # Tạo figure với kích thước vừa phải
    fig = plt.figure(figsize=(15, 10))
    
    # Đặt vị trí cửa sổ ở 4 góc màn hình
    positions = [(50, 50), (800, 50), (50, 500), (800, 500)]
    if figure_position < len(positions):
        try:
            mngr = fig.canvas.manager
            if hasattr(mngr, 'window'):
                mngr.window.wm_geometry(f"+{positions[figure_position][0]}+{positions[figure_position][1]}")
        except:
            pass  # Ignore nếu không thể set window position
    
    # Tạo 6 subplots
    axes = fig.subplots(6, 1)
    
    # Tính thời gian cho audio và features
    sr = 16000
    hop_size = thresholds['frame_shift']
    time_audio = np.arange(len(audio)) / sr
    time_features = np.arange(len(ste)) * hop_size
    
    # Tạo title chính
    clean_name = filename.replace('.wav', '').replace('_', ' ').title()
    fig.suptitle(f'SUV Classification - {clean_name}', fontsize=14, fontweight='bold')
    
    # 1. Tín hiệu âm thanh gốc
    axes[0].plot(time_audio, audio, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_title('Tín hiệu âm thanh gốc', fontsize=10)
    axes[0].set_ylabel('Amplitude', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Short-time Energy (STE) với ngưỡng
    axes[1].plot(time_features, ste, 'g-', linewidth=1.5)
    axes[1].axhline(y=thresholds['ste_threshold'], color='red', linestyle='--', alpha=0.8, 
                   label=f"STE Threshold={thresholds['ste_threshold']:.3f}")
    axes[1].set_title('Short-time Energy (STE) - Phân biệt Speech vs Silence', fontsize=10)
    axes[1].set_ylabel('STE', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Zero Crossing Rate (ZCR) với ngưỡng
    axes[2].plot(time_features, zcr, 'm-', linewidth=1.5)
    axes[2].axhline(y=thresholds['zcr_threshold'], color='red', linestyle='--', alpha=0.8,
                   label=f"ZCR Threshold={thresholds['zcr_threshold']:.3f}")
    axes[2].set_title('Zero Crossing Rate (ZCR) - Phân biệt Voiced vs Unvoiced', fontsize=10)
    axes[2].set_ylabel('ZCR', fontsize=9)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # 4. Spectrum Tilt (ST) với ngưỡng
    if len(st) > 0:
        axes[3].plot(time_features[:len(st)], st, 'orange', linewidth=1.5)
        axes[3].axhline(y=thresholds['st_threshold'], color='red', linestyle='--', alpha=0.8,
                       label=f"ST Threshold={thresholds['st_threshold']:.3f}")
        axes[3].set_title('Spectrum Tilt (ST) - SUVDA Algorithm', fontsize=10)
        axes[3].set_ylabel('ST', fontsize=9)
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)
    
    # 5. Dự đoán Raw (trước khi làm mịn)
    axes[4].plot(time_features[:len(predictions)], predictions, 'b-', 
                linewidth=2, alpha=0.8, label='Raw Prediction')
    axes[4].set_title('Kết quả phân loại Raw (trước khi làm mịn)', fontsize=10)
    axes[4].set_ylabel('Label', fontsize=9)
    axes[4].set_ylim(-0.5, 2.5)
    axes[4].set_yticks([0, 1, 2])
    axes[4].set_yticklabels(['Silence', 'Voiced', 'Unvoiced'], fontsize=8)
    axes[4].legend(fontsize=8)
    axes[4].grid(True, alpha=0.3)
    
    # 6. Kết quả cuối cùng (đã làm mịn)
    axes[5].plot(time_features[:len(smoothed_predictions)], smoothed_predictions, 'r-', 
                linewidth=2, label='Final Prediction')
    axes[5].set_title('Kết quả phân loại cuối cùng (đã làm mịn)', fontsize=10, fontweight='bold')
    axes[5].set_xlabel('Thời gian (s)', fontsize=9)
    axes[5].set_ylabel('Label', fontsize=9)
    axes[5].set_ylim(-0.5, 2.5)
    axes[5].set_yticks([0, 1, 2])
    axes[5].set_yticklabels(['Silence', 'Voiced', 'Unvoiced'], fontsize=8)
    axes[5].legend(fontsize=8)
    axes[5].grid(True, alpha=0.3)
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu figure
    save_path = os.path.join(results_dir, f"{filename.replace('.wav', '')}_result.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"    Đã lưu biểu đồ: {save_path}")


def plot_optimization_summary(optimization_results, results_dir):
    """
    Vẽ biểu đồ tóm tắt quá trình tối ưu ngưỡng
    
    Args:
        optimization_results: Kết quả tối ưu
        results_dir: Thư mục lưu kết quả
    """
    if not optimization_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Kết quả tối ưu ngưỡng SUV Classification', fontsize=14, fontweight='bold')
    
    # Subplot 1: Histogram của STE features theo class
    if 'ste_features' in optimization_results and 'labels' in optimization_results:
        ste_features = optimization_results['ste_features']
        labels = optimization_results['labels']
        
        silence_ste = [ste for ste, label in zip(ste_features, labels) if label == 0]
        voiced_ste = [ste for ste, label in zip(ste_features, labels) if label == 1]
        unvoiced_ste = [ste for ste, label in zip(ste_features, labels) if label == 2]
        
        axes[0,0].hist(silence_ste, bins=30, alpha=0.7, label='Silence', color='blue')
        axes[0,0].hist(voiced_ste, bins=30, alpha=0.7, label='Voiced', color='green')
        axes[0,0].hist(unvoiced_ste, bins=30, alpha=0.7, label='Unvoiced', color='red')
        axes[0,0].axvline(x=optimization_results['ste_threshold'], color='black', 
                         linestyle='--', label='Optimal Threshold')
        axes[0,0].set_title('STE Distribution by Class')
        axes[0,0].set_xlabel('STE Value')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    
    # Subplot 2: Histogram của ZCR features theo class
    if 'zcr_features' in optimization_results:
        zcr_features = optimization_results['zcr_features']
        
        silence_zcr = [zcr for zcr, label in zip(zcr_features, labels) if label == 0]
        voiced_zcr = [zcr for zcr, label in zip(zcr_features, labels) if label == 1]
        unvoiced_zcr = [zcr for zcr, label in zip(zcr_features, labels) if label == 2]
        
        axes[0,1].hist(silence_zcr, bins=30, alpha=0.7, label='Silence', color='blue')
        axes[0,1].hist(voiced_zcr, bins=30, alpha=0.7, label='Voiced', color='green')
        axes[0,1].hist(unvoiced_zcr, bins=30, alpha=0.7, label='Unvoiced', color='red')
        axes[0,1].axvline(x=optimization_results['zcr_threshold'], color='black', 
                         linestyle='--', label='Optimal Threshold')
        axes[0,1].set_title('ZCR Distribution by Class')
        axes[0,1].set_xlabel('ZCR Value')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Subplot 3: Histogram của ST features theo class
    if 'st_features' in optimization_results:
        st_features = optimization_results['st_features']
        
        silence_st = [st for st, label in zip(st_features, labels) if label == 0]
        voiced_st = [st for st, label in zip(st_features, labels) if label == 1]
        unvoiced_st = [st for st, label in zip(st_features, labels) if label == 2]
        
        axes[1,0].hist(silence_st, bins=30, alpha=0.7, label='Silence', color='blue')
        axes[1,0].hist(voiced_st, bins=30, alpha=0.7, label='Voiced', color='green')
        axes[1,0].hist(unvoiced_st, bins=30, alpha=0.7, label='Unvoiced', color='red')
        axes[1,0].axvline(x=optimization_results['st_threshold'], color='black', 
                         linestyle='--', label='Optimal Threshold')
        axes[1,0].set_title('ST Distribution by Class')
        axes[1,0].set_xlabel('ST Value')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Subplot 4: Class distribution
    if 'labels' in optimization_results:
        class_counts = [
            sum(1 for label in labels if label == 0),  # Silence
            sum(1 for label in labels if label == 1),  # Voiced
            sum(1 for label in labels if label == 2)   # Unvoiced
        ]
        class_names = ['Silence', 'Voiced', 'Unvoiced']
        colors = ['blue', 'green', 'red']
        
        bars = axes[1,1].bar(class_names, class_counts, color=colors, alpha=0.7)
        axes[1,1].set_title('Class Distribution in Training Data')
        axes[1,1].set_ylabel('Number of Frames')
        
        # Thêm số liệu lên bars
        for bar, count in zip(bars, class_counts):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{count}\n({count/sum(class_counts)*100:.1f}%)',
                          ha='center', va='bottom')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Lưu figure
    save_path = os.path.join(results_dir, "optimization_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Đã lưu biểu đồ tóm tắt tối ưu: {save_path}")


def _compute_accuracy(y_true, y_pred):
    """
    Hàm hỗ trợ tính accuracy
    
    Args:
        y_true: Nhãn thực
        y_pred: Nhãn dự đoán
        
    Returns:
        float: Accuracy
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)
