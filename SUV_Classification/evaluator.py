import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os

class SUVEvaluator:
    """
    Lớp đánh giá hiệu suất phân loại SUV
    """
    
    def __init__(self, sr=16000, hop_size=0.01):
        """
        Khởi tạo evaluator
        
        Args:
            sr (int): Tần số lấy mẫu
            hop_size (float): Bước nhảy khung (giây)
        """
        self.sr = sr
        self.hop_size = hop_size
        
    def segments_to_boundaries(self, segments: List[Dict]) -> List[float]:
        """
        Chuyển đổi segments thành danh sách boundaries
        
        Args:
            segments: Danh sách các segment
            
        Returns:
            List[float]: Danh sách thời điểm biên giới
        """
        boundaries = []
        for segment in segments:
            boundaries.append(segment['start'])
        # Thêm điểm cuối của segment cuối cùng
        if segments:
            boundaries.append(segments[-1]['end'])
        return sorted(list(set(boundaries)))  # Loại bỏ trùng lặp và sắp xếp
    
    def predictions_to_boundaries(self, predictions: np.ndarray) -> List[float]:
        """
        Chuyển đổi dự đoán frame thành boundaries
        
        Args:
            predictions: Mảng dự đoán cho mỗi frame
            
        Returns:
            List[float]: Danh sách thời điểm biên giới
        """
        boundaries = [0.0]  # Bắt đầu từ 0
        
        for i in range(1, len(predictions)):
            if predictions[i] != predictions[i-1]:
                # Có sự thay đổi label -> tạo boundary
                boundary_time = i * self.hop_size
                boundaries.append(boundary_time)
        
        # Thêm boundary cuối
        end_time = len(predictions) * self.hop_size
        boundaries.append(end_time)
        
        return boundaries
    
    def compute_boundary_error(self, true_boundaries: List[float], 
                             pred_boundaries: List[float]) -> Dict:
        """
        Tính sai số giữa boundaries thực và dự đoán
        
        Args:
            true_boundaries: Boundaries thực
            pred_boundaries: Boundaries dự đoán
            
        Returns:
            Dict: Các metrics đánh giá
        """
        # Tính sai số cho mỗi boundary dự đoán
        errors = []
        
        for pred_boundary in pred_boundaries[1:-1]:  # Bỏ boundary đầu và cuối
            # Tìm boundary thực gần nhất
            min_distance = float('inf')
            for true_boundary in true_boundaries[1:-1]:
                distance = abs(pred_boundary - true_boundary)
                if distance < min_distance:
                    min_distance = distance
            
            errors.append(min_distance)
        
        if len(errors) == 0:
            return {'mae': 0, 'rmse': 0, 'num_boundaries': 0}
        
        # Tính MAE và RMSE
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        
        return {
            'mae': mae,
            'rmse': rmse,
            'num_boundaries': len(errors),
            'errors': errors
        }
    
    def compute_frame_accuracy(self, true_labels: np.ndarray, 
                              pred_labels: np.ndarray) -> Dict:
        """
        Tính độ chính xác theo frame
        
        Args:
            true_labels: Nhãn thực
            pred_labels: Nhãn dự đoán
            
        Returns:
            Dict: Các metrics độ chính xác
        """
        min_length = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_length]
        pred_labels = pred_labels[:min_length]
        
        # Tính accuracy tổng thể
        overall_accuracy = np.mean(true_labels == pred_labels)
        
        # Tính accuracy cho từng class
        class_accuracies = {}
        for class_id, class_name in [(0, 'silence'), (1, 'voiced'), (2, 'unvoiced')]:
            mask = true_labels == class_id
            if np.sum(mask) > 0:
                class_acc = np.mean(pred_labels[mask] == class_id)
                class_accuracies[class_name] = class_acc
            else:
                class_accuracies[class_name] = 0.0
        
        return {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'total_frames': min_length
        }
    
    def plot_results(self, audio: np.ndarray, ste: np.ndarray, zcr: np.ndarray,
                    true_labels: np.ndarray, pred_labels: np.ndarray,
                    true_boundaries: List[float], pred_boundaries: List[float],
                    title: str, save_path: str, st: np.ndarray = None):
        """
        Vẽ kết quả phân loại SUVDA
        
        Args:
            audio: Tín hiệu âm thanh
            ste: Short-time Energy
            zcr: Zero Crossing Rate
            true_labels: Nhãn thực
            pred_labels: Nhãn dự đoán
            true_boundaries: Boundaries thực
            pred_boundaries: Boundaries dự đoán
            title: Tiêu đề biểu đồ
            save_path: Đường dẫn lưu hình
            st: Spectrum Tilt (optional)
        """
        # Điều chỉnh số subplot dựa trên có ST hay không
        num_plots = 6 if st is not None else 5
        fig, axes = plt.subplots(num_plots, 1, figsize=(15, 14 if st is not None else 12))
        
        # Thời gian cho audio
        time_audio = np.arange(len(audio)) / self.sr
        
        # Thời gian cho features
        time_features = np.arange(len(ste)) * self.hop_size
        
        # 1. Tín hiệu âm thanh gốc
        axes[0].plot(time_audio, audio, 'b-', alpha=0.7)
        axes[0].set_title(f'{title} - Tín hiệu âm thanh gốc')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Vẽ boundaries thực (đỏ) và dự đoán (xanh)
        for boundary in true_boundaries:
            axes[0].axvline(x=boundary, color='red', linestyle='-', alpha=0.7, label='Ground Truth')
        for boundary in pred_boundaries:
            axes[0].axvline(x=boundary, color='blue', linestyle='--', alpha=0.7, label='Predicted')
        
        # 2. STE
        axes[1].plot(time_features, ste, 'g-', linewidth=1.5)
        axes[1].set_title('Short-time Energy (STE)')
        axes[1].set_ylabel('Normalized STE')
        axes[1].grid(True, alpha=0.3)
        
        # 3. ZCR
        axes[2].plot(time_features, zcr, 'm-', linewidth=1.5)
        axes[2].set_title('Zero Crossing Rate (ZCR)')
        axes[2].set_ylabel('ZCR')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Spectrum Tilt (nếu có)
        plot_idx = 3
        if st is not None:
            axes[plot_idx].plot(time_features[:len(st)], st, 'orange', linewidth=1.5)
            axes[plot_idx].set_title('Spectrum Tilt (ST) - SUVDA')
            axes[plot_idx].set_ylabel('ST')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Threshold=0.7')
            axes[plot_idx].legend()
            plot_idx += 1
        
        # Nhãn thực
        axes[plot_idx].plot(time_features[:len(true_labels)], true_labels, 'r-', linewidth=2, label='Ground Truth')
        axes[plot_idx].set_title('Nhãn thực (Ground Truth)')
        axes[plot_idx].set_ylabel('Label (0:sil, 1:v, 2:uv)')
        axes[plot_idx].set_ylim(-0.5, 2.5)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
        
        # So sánh nhãn thực vs dự đoán
        min_len = min(len(true_labels), len(pred_labels))
        axes[plot_idx].plot(time_features[:min_len], true_labels[:min_len], 'r-', linewidth=2, label='Ground Truth')
        axes[plot_idx].plot(time_features[:min_len], pred_labels[:min_len], 'b--', linewidth=2, alpha=0.7, label='Predicted')
        axes[plot_idx].set_title('So sánh: Ground Truth vs Predicted (SUVDA)')
        axes[plot_idx].set_xlabel('Thời gian (s)')
        axes[plot_idx].set_ylabel('Label (0:sil, 1:v, 2:uv)')
        axes[plot_idx].set_ylim(-0.5, 2.5)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Vẽ boundaries cho tất cả subplot
        for i in range(num_plots):
            for boundary in true_boundaries:
                axes[i].axvline(x=boundary, color='red', linestyle='-', alpha=0.3)
            for boundary in pred_boundaries:
                axes[i].axvline(x=boundary, color='blue', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Đã lưu kết quả vào: {save_path}")
    
    def generate_report(self, results: List[Dict]) -> str:
        """
        Tạo báo cáo tổng hợp
        
        Args:
            results: Danh sách kết quả đánh giá
            
        Returns:
            str: Nội dung báo cáo
        """
        report = "=== BÁO CÁO ĐÁNH GIÁ PHÂN LOẠI SUV ===\\n\\n"
        
        total_mae = []
        total_rmse = []
        total_accuracy = []
        
        for i, result in enumerate(results):
            filename = result.get('filename', f'File {i+1}')
            boundary_metrics = result.get('boundary_metrics', {})
            frame_metrics = result.get('frame_metrics', {})
            
            report += f"File: {filename}\\n"
            report += f"  Boundary Error - MAE: {boundary_metrics.get('mae', 0):.4f}s, RMSE: {boundary_metrics.get('rmse', 0):.4f}s\\n"
            report += f"  Frame Accuracy: {frame_metrics.get('overall_accuracy', 0):.4f}\\n"
            
            class_acc = frame_metrics.get('class_accuracies', {})
            report += f"  Class Accuracies - Silence: {class_acc.get('silence', 0):.4f}, "
            report += f"Voiced: {class_acc.get('voiced', 0):.4f}, Unvoiced: {class_acc.get('unvoiced', 0):.4f}\\n\\n"
            
            if boundary_metrics.get('mae') is not None:
                total_mae.append(boundary_metrics['mae'])
            if boundary_metrics.get('rmse') is not None:
                total_rmse.append(boundary_metrics['rmse'])
            if frame_metrics.get('overall_accuracy') is not None:
                total_accuracy.append(frame_metrics['overall_accuracy'])
        
        # Tính trung bình
        if total_mae:
            report += f"TRUNG BÌNH:\\n"
            report += f"  Boundary Error - MAE: {np.mean(total_mae):.4f}s, RMSE: {np.mean(total_rmse):.4f}s\\n"
            report += f"  Frame Accuracy: {np.mean(total_accuracy):.4f}\\n"
        
        return report
