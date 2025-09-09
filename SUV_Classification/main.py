#!/usr/bin/env python3
"""
HỆ THỐNG TỐI ỬU NGƯỠNG VÀ PHÂN LOẠI SUV TỰ ĐỘNG
Chương trình tối ưu ngưỡng sử dụng ground truth từ file .lab
và so sánh kết quả phân loại với ground truth
Chỉ sử dụng các hàm tự viết và built-in functions của Python/Numpy
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from threshold_optimizer import find_training_files, optimize_thresholds_with_ground_truth, save_optimization_results
from evaluator import find_test_files_with_labels, demo_classification_with_ground_truth

def main():
    """
    CHƯƠNG TRÌNH CHÍNH: TỐI ỬU NGƯỠNG VÀ DEMO PHÂN LOẠI SUV
    1. Tối ưu ngưỡng tự động sử dụng ground truth từ file .lab
    2. Demo phân loại trên 4 file test với ngưỡng tối ưu
    3. So sánh kết quả với ground truth và hiển thị trên 4 figure
    """
    print("=== HỆ THỐNG TỐI ỬU NGƯỠNG VÀ PHÂN LOẠI SUV ===\n")
    
    # Cấu hình đường dẫn
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Thư mục dữ liệu
    training_data_dir = os.path.join(project_root, "thigiuaki", "TinHieuHuanLuyen")
    test_data_dir = os.path.join(project_root, "thigiuaki", "TinHieuKiemThu")
    results_dir = os.path.join(project_root, "results")
    
    # Tạo thư mục kết quả
    os.makedirs(results_dir, exist_ok=True)
    
    # BƯỚC 1: TỐI ỬU NGƯỠNG VỚI GROUND TRUTH
    print("=== BƯỚC 1: TỐI ỬU NGƯỠNG TỰ ĐỘNG ===")
    
    training_files = find_training_files(training_data_dir)
    
    if len(training_files) == 0:
        print("Không tìm thấy file training với ground truth!")
        print("Sử dụng ngưỡng mặc định từ bài báo SUVDA...")
        
        best_thresholds = {
            'ste_threshold': -1.5,
            'zcr_threshold': 0.3,
            'st_threshold': 0.7,
            'frame_length': 0.025,
            'frame_shift': 0.010
        }
    else:
        print(f"Tìm thấy {len(training_files)} file training với ground truth:")
        for wav_path, lab_path in training_files:
            print(f"  - {os.path.basename(wav_path)}")
        print()
        
        # Thực hiện tối ưu ngưỡng
        best_thresholds = optimize_thresholds_with_ground_truth(training_files)
        
        # Lưu kết quả tối ưu
        save_optimization_results(best_thresholds, results_dir)
    
    # BƯỚC 2: DEMO PHÂN LOẠI VỚI NGƯỠNG TỐI ỬU
    print("\n=== BƯỚC 2: DEMO PHÂN LOẠI VỚI NGƯỠNG TỐI ỬU ===")
    
    test_files = find_test_files_with_labels(test_data_dir)
    
    if len(test_files) == 0:
        print("Không tìm thấy file test với ground truth!")
        return
    
    print(f"Tìm thấy {len(test_files)} file test với ground truth:")
    for wav_path, lab_path in test_files:
        print(f"  - {os.path.basename(wav_path)}")
    print()
    
    # Demo phân loại với ngưỡng tối ưu và so sánh với ground truth
    demo_classification_with_ground_truth(test_files, best_thresholds, results_dir)
    
    print("\n🎯 HOÀN THÀNH!")
    print("📄 Kết quả đã được lưu trong thư mục results/")
    print("📊 Các figure hiển thị so sánh với ground truth")


if __name__ == "__main__":
    main()
