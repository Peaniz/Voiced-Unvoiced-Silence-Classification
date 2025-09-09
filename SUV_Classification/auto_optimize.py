#!/usr/bin/env python3
"""
Script tự động tối ưu ngưỡng cho SUV Classification
"""

import os
import sys
import glob
import json
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from threshold_optimizer import ThresholdOptimizer
from suv_classifier import SUVClassifier
from evaluator import SUVEvaluator
from audio_analyzer import AudioAnalyzer

def main():
    """
    Chạy tối ưu ngưỡng tự động
    """
    print("=== HỆ THỐNG TỐI ỬU NGƯỠNG TỰ ĐỘNG ===\\n")
    
    # Cấu hình đường dẫn
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Thư mục dữ liệu training
    training_data_dir = os.path.join(project_root, "Thi giữa kỳ", "TinHieuHuanLuyen")
    
    # Thư mục kết quả
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Kiểm tra thư mục dữ liệu
    if not os.path.exists(training_data_dir):
        print(f"Không tìm thấy thư mục dữ liệu: {training_data_dir}")
        return
    
    # Lấy danh sách file training
    training_files = []
    wav_files = glob.glob(os.path.join(training_data_dir, "*.wav"))
    
    for wav_file in wav_files:
        lab_file = wav_file.replace('.wav', '.lab')
        if os.path.exists(lab_file):
            training_files.append((wav_file, lab_file))
        else:
            print(f"Không tìm thấy file label cho {wav_file}")
    
    if len(training_files) == 0:
        print("Không tìm thấy file training nào!")
        return
        
    print(f"Tìm thấy {len(training_files)} file training:")
    for wav_path, lab_path in training_files:
        print(f"  - {os.path.basename(wav_path)}")
    print()
    
    # Khởi tạo optimizer
    optimizer = ThresholdOptimizer()
    
    # === BƯỚC 1: DYNAMIC THRESHOLD OPTIMIZATION ===
    print("BƯỚC 1: DYNAMIC THRESHOLD OPTIMIZATION (UNSUPERVISED)\\n")
    
    # Chỉ lấy wav files, không cần lab files - với validation
    wav_files = []
    for wav_path, _ in training_files:
        if os.path.exists(wav_path) and wav_path.endswith('.wav'):
            wav_files.append(wav_path)
        else:
            print(f"Warning: File not found or invalid: {wav_path}")
    
    if len(wav_files) == 0:
        print("\\n❌ ERROR: No valid audio files found!")
        return
        
    print(f"Found {len(wav_files)} valid audio files for dynamic threshold optimization")
    
    best_params = optimizer.optimize_dynamic_thresholds(
        training_files=wav_files,
        verbose=True
    )
    
    # KIỂM TRA KẾT QUẢ OPTIMIZATION
    if best_params is None:
        print("\\n❌ OPTIMIZATION FAILED - No valid parameters found!")
        print("This could be due to:")
        print("  • Invalid audio files")
        print("  • Insufficient data")
        print("  • Computation errors")
        return
    
    # Lưu kết quả tối ưu
    best_params_file = os.path.join(results_dir, "best_thresholds.json")
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    
    print(f"\\nĐã lưu ngưỡng tối ưu vào: {best_params_file}")
    
    # === BƯỚC 2: DEMO DYNAMIC THRESHOLDS ===
    print("\\n" + "="*60)
    print("BƯỚC 2: DEMO DYNAMIC THRESHOLDS TRÊN TẤT CẢ FILE")
    print("="*60)
    
    # Tạo classifier với ngưỡng tối ưu
    optimal_classifier = SUVClassifier(
        frame_length=best_params['frame_length'],
        frame_shift=best_params['frame_shift'],
        sr=16000
    )
    
    # Set ngưỡng tối ưu (bao gồm ST thresholds mới)
    optimal_classifier.ste_thresholds = {
        'speech_silence': best_params.get('ste_speech_silence', best_params.get('energy_threshold', 0)),
        'voiced_unvoiced': best_params.get('ste_voiced_unvoiced', 0)
    }
    optimal_classifier.zcr_thresholds = {
        'speech_silence': best_params.get('zcr_speech_silence', 0), 
        'voiced_unvoiced': best_params.get('zcr_voiced_unvoiced', best_params.get('zcr_threshold', 0.5))
    }
    # Khởi tạo ST thresholds cho SUVDA
    optimal_classifier.st_thresholds = {
        'speech_silence': 0,
        'voiced_unvoiced': best_params.get('st_voiced_unvoiced', best_params.get('st_threshold', 0.7))
    }
    optimal_classifier.trained = True
    
    # Đánh giá từng file
    evaluator = SUVEvaluator(sr=16000, hop_size=best_params['frame_shift'])
    analyzer = AudioAnalyzer(
        frame_length=best_params['frame_length'],
        frame_shift=best_params['frame_shift'],
        sr=16000
    )
    
    evaluation_results = []
    
    for i, (wav_path, _) in enumerate(training_files):
        filename = os.path.basename(wav_path)
        print(f"\\nDemo file {i+1}/{len(training_files)}: {filename}")
        
        # Phân loại với dynamic thresholds
        result = optimal_classifier.classify(wav_path)
        if len(result) == 5:  # Có ST
            audio, ste, zcr, st, predictions = result
        else:  # Fallback
            audio, ste, zcr, predictions = result
            st = None
        smoothed_predictions = optimal_classifier.smooth_predictions(predictions, min_segment_length=30)
        
        # Thống kê kết quả classification (không cần ground truth)
        unique, counts = np.unique(smoothed_predictions, return_counts=True)
        total_frames = len(smoothed_predictions)
        
        print(f"  Total frames: {total_frames}")
        for label, count in zip(unique, counts):
            label_name = ["Silence", "Voiced", "Unvoiced"][label]
            percentage = (count / total_frames) * 100
            print(f"  {label_name}: {count} frames ({percentage:.1f}%)")
        
        # Hiển thị feature statistics
        print(f"  Feature ranges:")
        print(f"    STE: [{np.min(ste):.4f}, {np.max(ste):.4f}], mean: {np.mean(ste):.4f}")
        print(f"    ZCR: [{np.min(zcr):.4f}, {np.max(zcr):.4f}], mean: {np.mean(zcr):.4f}")
        if st is not None:
            print(f"    ST: [{np.min(st):.4f}, {np.max(st):.4f}], mean: {np.mean(st):.4f}")
        
        # Tính predicted boundaries (chỉ để demo)
        pred_boundaries = evaluator.predictions_to_boundaries(smoothed_predictions)
        print(f"  Predicted segments: {len(pred_boundaries)//2 if len(pred_boundaries) > 0 else 0}")
        
    
    # === TẠO BÁO CÁO TỔNG HỢP ===
    print("\\n" + "="*60)
    print("BÁO CÁO TỔNG HỢP")
    print("="*60)
    
    # Demo hoàn thành - không cần accuracy metrics
    
    print(f"\\nDYNAMIC THRESHOLDS SUMMARY:")
    print(f"Separation Score: {best_params['separation_score']:.4f}")
    print(f"Optimal W parameters: STE={best_params['W_STE']}, ZCR={best_params['W_ZCR']}, ST={best_params['W_ST']}")
    print(f"Files processed: {len(training_files)}")
    
    # Lưu báo cáo dynamic thresholds
    report_file = os.path.join(results_dir, "dynamic_threshold_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== DYNAMIC THRESHOLD OPTIMIZATION REPORT ===\\n\\n")
        f.write("Approach: UNSUPERVISED (không cần ground truth)\\n")
        f.write("Công thức: T = (W × M1 + M2) / (W + 1)\\n")
        f.write("M1, M2: vị trí của 2 local maxima trong histogram\\n\\n")
        
        f.write("THAM SỐ TỐI ỬU:\\n")
        f.write(f"Frame Length: {best_params['frame_length']*1000:.0f}ms\\n")
        f.write(f"Frame Shift: {best_params['frame_shift']*1000:.0f}ms\\n")
        f.write(f"W_STE: {best_params['W_STE']}\\n")
        f.write(f"W_ZCR: {best_params['W_ZCR']}\\n")
        f.write(f"W_ST: {best_params['W_ST']}\\n\\n")
        
        f.write("DYNAMIC THRESHOLDS COMPUTED:\\n")
        f.write(f"T_STE (Energy): {best_params['energy_threshold']:.6f}\\n")
        f.write(f"T_ZCR: {best_params['zcr_threshold']:.6f}\\n")
        f.write(f"T_ST (Spectrum Tilt): {best_params['st_threshold']:.6f}\\n\\n")
        
        f.write(f"SEPARATION SCORE: {best_params['separation_score']:.4f}\\n")
        f.write("(Unsupervised metric - higher = better feature separation)\\n")
    
    print(f"\\nĐã lưu báo cáo vào: {report_file}")
    
    # Lấy ngưỡng dynamic
    energy_thresh = best_params['energy_threshold']
    zcr_thresh = best_params['zcr_threshold']
    st_thresh = best_params['st_threshold']
    
    print(f"\\nDYNAMIC THRESHOLDS RESULT:")
    print(f"   Frame: {best_params['frame_length']*1000:.0f}ms/{best_params['frame_shift']*1000:.0f}ms")
    print(f"   W parameters: STE={best_params['W_STE']}, ZCR={best_params['W_ZCR']}, ST={best_params['W_ST']}")
    print(f"   T_STE (Energy): {energy_thresh:.6f}")
    print(f"   T_ZCR: {zcr_thresh:.6f}")  
    print(f"   T_ST (Spectrum Tilt): {st_thresh:.6f}")
    
    print(f"\\nSUVDA LOGIC (DYNAMIC THRESHOLDS - UNSUPERVISED):")
    print(f"   SILENCE: STE < {energy_thresh:.4f} AND ZCR < {zcr_thresh:.4f}")
    print(f"   VOICED: STE >= {energy_thresh:.4f} AND ST > {st_thresh:.4f} AND ZCR < {zcr_thresh:.4f}")
    print(f"   UNVOICED: STE >= {energy_thresh:.4f} AND ST < {st_thresh:.4f} AND ZCR > {zcr_thresh:.4f}")
    
    print("\\n🎯 DYNAMIC THRESHOLD OPTIMIZATION COMPLETED!")
    print("📄 Files saved:")
    print(f"   • {best_params_file}")
    print(f"   • {report_file}")
    print("✅ No ground truth required - fully unsupervised approach!")

if __name__ == "__main__":
    main()
