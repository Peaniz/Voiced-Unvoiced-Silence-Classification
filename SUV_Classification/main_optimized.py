#!/usr/bin/env python3
"""
Script chính sử dụng ngưỡng đã được tối ưu
"""

import os
import sys
import glob
import json
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from suv_classifier import SUVClassifier
from evaluator import SUVEvaluator
from audio_analyzer import AudioAnalyzer

def load_optimized_thresholds(results_dir: str):
    """
    Load ngưỡng đã được tối ưu từ file
    """
    thresholds_file = os.path.join(results_dir, "best_thresholds.json")
    
    if os.path.exists(thresholds_file):
        with open(thresholds_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"Không tìm thấy file ngưỡng tối ưu: {thresholds_file}")
        print("Hãy chạy auto_optimize.py trước để tìm ngưỡng tối ưu!")
        return None

def main():
    """
    Chạy phân loại SUV với ngưỡng đã tối ưu
    """
    print("=== SUV CLASSIFICATION VỚI NGƯỠNG TỐI ỬU ===\\n")
    
    # Cấu hình đường dẫn
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Thư mục dữ liệu
    training_data_dir = os.path.join(project_root, "Thi giữa kỳ", "TinHieuHuanLuyen")
    
    # Thư mục kết quả
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load ngưỡng tối ưu
    optimized_params = load_optimized_thresholds(results_dir)
    if optimized_params is None:
        print("Chạy chế độ tối ưu mặc định...")
        # Fallback: sử dụng main.py gốc
        from main import main as original_main
        original_main()
        return
    
    print("SU DUNG NGUONG DA TOI UU:")
    print(f"   Frame: {optimized_params['frame_length']*1000:.0f}ms/{optimized_params['frame_shift']*1000:.0f}ms")
    
    # Lấy ngưỡng (có thể là cấu trúc cũ hoặc mới)
    energy_thresh = optimized_params.get('energy_threshold', optimized_params.get('ste_speech_silence', 0))
    zcr_thresh = optimized_params.get('zcr_threshold', optimized_params.get('zcr_voiced_unvoiced', 0))
    st_thresh = optimized_params.get('st_threshold', optimized_params.get('st_voiced_unvoiced', 0.7))
    
    print(f"   Energy Threshold: {energy_thresh:.6f} (STE cho silence vs speech)")
    print(f"   ZCR Threshold: {zcr_thresh:.6f} (ZCR cho voiced vs unvoiced)")
    print(f"   ST Threshold: {st_thresh:.6f} (Spectrum Tilt cho voiced vs unvoiced)")
    print(f"   Optimization Score: {optimized_params['score']:.4f}")
    
    print(f"\\nLOGIC SUVDA (FIXED - TU BAI BAO, CHI TOI UU NGUONG):")
    print(f"   SILENCE: STE < {energy_thresh:.4f} AND ZCR < {zcr_thresh:.4f}")
    print(f"   VOICED: STE >= {energy_thresh:.4f} AND ST > {st_thresh:.4f} AND ZCR < {zcr_thresh:.4f}")
    print(f"   UNVOICED: STE >= {energy_thresh:.4f} AND ST < {st_thresh:.4f} AND ZCR > {zcr_thresh:.4f}\\n")
    
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
    
    if len(training_files) == 0:
        print("Không tìm thấy file training nào!")
        return
    
    print(f"Tìm thấy {len(training_files)} file training:")
    for wav_path, lab_path in training_files:
        print(f"  - {os.path.basename(wav_path)}")
    print()
    
    # Khởi tạo classifier với params tối ưu
    classifier = SUVClassifier(
        frame_length=optimized_params['frame_length'],
        frame_shift=optimized_params['frame_shift'],
        sr=16000
    )
    
    # Set ngưỡng tối ưu SUVDA (3 ngưỡng)
    energy_thresh = optimized_params.get('energy_threshold', optimized_params.get('ste_speech_silence', 0))
    zcr_thresh = optimized_params.get('zcr_threshold', optimized_params.get('zcr_voiced_unvoiced', 0))
    st_thresh = optimized_params.get('st_threshold', optimized_params.get('st_voiced_unvoiced', 0.7))
    
    classifier.ste_thresholds = {
        'speech_silence': energy_thresh,
        'voiced_unvoiced': 0  # Không dùng trong logic SUVDA
    }
    classifier.zcr_thresholds = {
        'speech_silence': 0,  # Không dùng trong logic SUVDA
        'voiced_unvoiced': zcr_thresh
    }
    # Khởi tạo ST thresholds (nếu chưa có)
    if not hasattr(classifier, 'st_thresholds'):
        classifier.st_thresholds = {}
    classifier.st_thresholds = {
        'speech_silence': 0,  # Không dùng trong logic SUVDA
        'voiced_unvoiced': st_thresh
    }
    classifier.trained = True  # Skip training vì đã có ngưỡng
    
    # Khởi tạo evaluator và analyzer với params tối ưu
    evaluator = SUVEvaluator(sr=16000, hop_size=optimized_params['frame_shift'])
    analyzer = AudioAnalyzer(
        frame_length=optimized_params['frame_length'],
        frame_shift=optimized_params['frame_shift'],
        sr=16000
    )
    
    print("=== ĐÁNH GIÁ VỚI NGƯỠNG TỐI ỬU ===")
    evaluation_results = []
    
    for i, (wav_path, lab_path) in enumerate(training_files):
        filename = os.path.basename(wav_path)
        print(f"\\nXử lý file {i+1}/{len(training_files)}: {filename}")
        
        # Phân loại với ngưỡng tối ưu (bao gồm ST)
        result = classifier.classify(wav_path)
        if len(result) == 5:  # Có ST
            audio, ste, zcr, st, predictions = result
        else:  # Fallback
            audio, ste, zcr, predictions = result
            st = None
        
        # Làm mịn kết quả
        smoothed_predictions = classifier.smooth_predictions(predictions, min_segment_length=30)
        
        # Load ground truth
        segments = analyzer.load_labels(lab_path)
        true_labels = analyzer.get_frame_labels(segments, len(audio))
        
        # Đảm bảo chiều dài khớp nhau
        min_length = min(len(true_labels), len(smoothed_predictions))
        true_labels = true_labels[:min_length]
        smoothed_predictions = smoothed_predictions[:min_length]
        
        # Tính toán boundaries
        true_boundaries = evaluator.segments_to_boundaries(segments)
        pred_boundaries = evaluator.predictions_to_boundaries(smoothed_predictions)
        
        # Đánh giá
        boundary_metrics = evaluator.compute_boundary_error(true_boundaries, pred_boundaries)
        frame_metrics = evaluator.compute_frame_accuracy(true_labels, smoothed_predictions)
        
        print(f"  Boundary Error - MAE: {boundary_metrics['mae']:.4f}s, RMSE: {boundary_metrics['rmse']:.4f}s")
        print(f"  Frame Accuracy: {frame_metrics['overall_accuracy']:.4f}")
        print(f"  Class Accuracies - Silence: {frame_metrics['class_accuracies']['silence']:.4f}, " + 
              f"Voiced: {frame_metrics['class_accuracies']['voiced']:.4f}, " +
              f"Unvoiced: {frame_metrics['class_accuracies']['unvoiced']:.4f}")
        
        # Lưu kết quả
        result = {
            'filename': filename,
            'boundary_metrics': boundary_metrics,
            'frame_metrics': frame_metrics
        }
        evaluation_results.append(result)
        
        # Vẽ và lưu biểu đồ (SUVDA với ST)
        plot_title = f"SUV Classification (SUVDA Optimized) - {filename.replace('.wav', '')}"
        plot_path = os.path.join(results_dir, f"{filename.replace('.wav', '')}_suvda_result.png")
        
        evaluator.plot_results(
            audio=audio,
            ste=ste,
            zcr=zcr,
            true_labels=true_labels,
            pred_labels=smoothed_predictions,
            true_boundaries=true_boundaries,
            pred_boundaries=pred_boundaries,
            title=plot_title,
            save_path=plot_path,
            st=st  # Thêm Spectrum Tilt
        )
    
    # Tạo báo cáo tổng hợp
    print("\\n=== BÁO CÁO KẾT QUẢ VỚI NGƯỠNG TỐI ỬU ===")
    
    # Thống kê tổng hợp
    accuracies = [r['frame_metrics']['overall_accuracy'] for r in evaluation_results]
    maes = [r['boundary_metrics']['mae'] for r in evaluation_results]
    rmses = [r['boundary_metrics']['rmse'] for r in evaluation_results]
    
    print(f"\\nTHỐNG KÊ HIỆU SUẤT:")
    print(f"Frame Accuracy - Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")
    print(f"Boundary MAE - Mean: {np.mean(maes):.4f}s, Std: {np.std(maes):.4f}s")
    print(f"Boundary RMSE - Mean: {np.mean(rmses):.4f}s, Std: {np.std(rmses):.4f}s")
    
    # So sánh với baseline (nếu có)
    baseline_file = os.path.join(results_dir, "evaluation_report.txt")
    if os.path.exists(baseline_file):
        print("\\n📊 SO SÁNH VỚI BASELINE:")
        print("   (Xem chi tiết trong file báo cáo)")
    
    # Tạo báo cáo
    report_content = evaluator.generate_report(evaluation_results)
    
    optimized_report_file = os.path.join(results_dir, "optimized_evaluation_report.txt")
    with open(optimized_report_file, 'w', encoding='utf-8') as f:
        f.write("=== BÁO CÁO SUV CLASSIFICATION VỚI NGƯỠNG TỐI ỬU ===\\n\\n")
        
        f.write("THAM SỐ TỐI ỬU SỬ DỤNG:\\n")
        f.write(f"Frame Length: {optimized_params['frame_length']*1000:.0f}ms\\n")
        f.write(f"Frame Shift: {optimized_params['frame_shift']*1000:.0f}ms\\n")
        
        energy_thresh = optimized_params.get('energy_threshold', optimized_params.get('ste_speech_silence', 0))
        zcr_thresh = optimized_params.get('zcr_threshold', optimized_params.get('zcr_voiced_unvoiced', 0))
        st_thresh = optimized_params.get('st_threshold', optimized_params.get('st_voiced_unvoiced', 0.7))
        
        f.write(f"🎯 Energy Threshold: {energy_thresh:.6f} (STE cho silence vs speech)\\n")
        f.write(f"🎯 ZCR Threshold: {zcr_thresh:.6f} (ZCR cho voiced vs unvoiced)\\n")
        f.write(f"🆕 ST Threshold: {st_thresh:.6f} (Spectrum Tilt cho voiced vs unvoiced)\\n")
        f.write(f"Optimization Score: {optimized_params['score']:.4f}\\n")
        
        f.write(f"\\nLOGIC SUVDA (3 ĐẶC TRƯNG):\\n")
        f.write(f"1. SILENCE: STE < {energy_thresh:.6f} AND ZCR < {zcr_thresh:.6f}\\n")
        f.write(f"2. VOICED: STE ≥ {energy_thresh:.6f} AND ST > {st_thresh:.6f} AND ZCR < {zcr_thresh:.6f}\\n")
        f.write(f"3. UNVOICED: STE ≥ {energy_thresh:.6f} AND ST < {st_thresh:.6f} AND ZCR > {zcr_thresh:.6f}\\n\\n")
        
        f.write(report_content)
    
    print(f"\\nĐã lưu báo cáo vào: {optimized_report_file}")
    print(f"\\n✅ HOÀN THÀNH ĐÁNH GIÁ VỚI NGƯỠNG TỐI ỬU!")
    print(f"📈 Frame Accuracy trung bình: {np.mean(accuracies):.4f}")
    print(f"📊 Tất cả kết quả trong thư mục: {results_dir}")

if __name__ == "__main__":
    main()
