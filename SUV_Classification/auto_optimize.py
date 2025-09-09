#!/usr/bin/env python3
"""
Script tự động tối ưu ngưỡng cho SUV Classification với ground truth
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
    Chạy tối ưu ngưỡng tự động với ground truth từ file lab
    """
    print("=== HỆ THỐNG TỐI ỬU NGƯỠNG TỰ ĐỘNG (SUPERVISED) ===\\n")
    
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
    
    # Lấy danh sách file training (cần cả wav và lab)
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
    
    # === TỐI ỬU NGƯỠNG BẰNG GRID SEARCH ===
    print("BƯỚC 1: GRID SEARCH OPTIMIZATION (SUPERVISED)\\n")
    
    best_params = optimizer.optimize_thresholds_grid_search(
        training_files=training_files,
        validation_split=0.3,
        verbose=True
    )
    
    # KIỂM TRA KẾT QUẢ OPTIMIZATION
    if not best_params or best_params.get('score', 0) == 0:
        print("\\n❌ OPTIMIZATION FAILED - No valid parameters found!")
        print("This could be due to:")
        print("  • Invalid audio/label files")
        print("  • Insufficient data")
        print("  • Computation errors")
        return
    
    # Lưu kết quả tối ưu
    best_params_file = os.path.join(results_dir, "best_thresholds.json")
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    
    print(f"\\nĐã lưu ngưỡng tối ưu vào: {best_params_file}")
    
    # === ĐÁNH GIÁ VỚI NGƯỠNG TỐI ỬU ===
    print("\\n" + "="*60)
    print("BƯỚC 2: ĐÁNH GIÁ VỚI NGƯỠNG TỐI ỬU")
    print("="*60)
    
    # Tạo classifier với ngưỡng tối ưu
    optimal_classifier = SUVClassifier(
        frame_length=best_params['frame_length'],
        frame_shift=best_params['frame_shift'],
        sr=16000
    )
    
    # Set ngưỡng tối ưu
    optimal_classifier.ste_thresholds = {
        'speech_silence': best_params.get('ste_speech_silence', best_params.get('energy_threshold', 0)),
        'voiced_unvoiced': best_params.get('ste_voiced_unvoiced', 0)
    }
    optimal_classifier.zcr_thresholds = {
        'speech_silence': best_params.get('zcr_speech_silence', 0), 
        'voiced_unvoiced': best_params.get('zcr_voiced_unvoiced', best_params.get('zcr_threshold', 0.5))
    }
    optimal_classifier.st_thresholds = {
        'speech_silence': best_params.get('st_speech_silence', 0),
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
    
    for i, (wav_path, lab_path) in enumerate(training_files):
        filename = os.path.basename(wav_path)
        print(f"\\nĐánh giá file {i+1}/{len(training_files)}: {filename}")
        
        # Phân loại với ngưỡng tối ưu
        result = optimal_classifier.classify(wav_path)
        if len(result) == 5:  # Có ST
            audio, ste, zcr, st, predictions = result
        else:  # Fallback
            audio, ste, zcr, predictions = result
            st = None
        smoothed_predictions = optimal_classifier.smooth_predictions(predictions, min_segment_length=30)
        
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
        
        # Vẽ và lưu biểu đồ
        plot_title = f"SUV Classification (Optimized) - {filename.replace('.wav', '')}"
        plot_path = os.path.join(results_dir, f"{filename.replace('.wav', '')}_optimized_result.png")
        
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
            st=st
        )
    
    # === TẠO BÁO CÁO TỔNG HỢP ===
    print("\\n" + "="*60)
    print("BÁO CÁO KẾT QUẢ VỚI NGƯỠNG TỐI ỬU")
    print("="*60)
    
    # Thống kê tổng hợp
    accuracies = [r['frame_metrics']['overall_accuracy'] for r in evaluation_results]
    maes = [r['boundary_metrics']['mae'] for r in evaluation_results]
    rmses = [r['boundary_metrics']['rmse'] for r in evaluation_results]
    
    print(f"\\nTHỐNG KÊ HIỆU SUẤT:")
    print(f"Frame Accuracy - Mean: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")
    print(f"Boundary MAE - Mean: {np.mean(maes):.4f}s, Std: {np.std(maes):.4f}s")
    print(f"Boundary RMSE - Mean: {np.mean(rmses):.4f}s, Std: {np.std(rmses):.4f}s")
    
    # Lưu báo cáo tối ưu
    report_file = os.path.join(results_dir, "threshold_optimization_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== THRESHOLD OPTIMIZATION REPORT ===\\n\\n")
        f.write("Approach: SUPERVISED (với ground truth từ file .lab)\\n")
        f.write("Method: Grid Search với Cross-Validation\\n\\n")
        
        f.write("THAM SỐ TỐI ỬU:\\n")
        f.write(f"Frame Length: {best_params['frame_length']*1000:.0f}ms\\n")
        f.write(f"Frame Shift: {best_params['frame_shift']*1000:.0f}ms\\n")
        
        energy_thresh = best_params.get('energy_threshold', best_params.get('ste_speech_silence', 0))
        zcr_thresh = best_params.get('zcr_threshold', best_params.get('zcr_voiced_unvoiced', 0))
        st_thresh = best_params.get('st_threshold', best_params.get('st_voiced_unvoiced', 0.7))
        
        f.write(f"Energy Threshold: {energy_thresh:.6f}\\n")
        f.write(f"ZCR Threshold: {zcr_thresh:.6f}\\n")
        f.write(f"ST Threshold: {st_thresh:.6f}\\n")
        f.write(f"Optimization Score: {best_params['score']:.4f}\\n\\n")
        
        f.write("HIỆU SUẤT:\\n")
        f.write(f"Frame Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\\n")
        f.write(f"Boundary MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}s\\n")
        f.write(f"Boundary RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}s\\n\\n")
        
        f.write(optimizer.get_optimization_report())
    
    print(f"\\nĐã lưu báo cáo vào: {report_file}")
    
    # Hiển thị ngưỡng cuối cùng
    energy_thresh = best_params.get('energy_threshold', best_params.get('ste_speech_silence', 0))
    zcr_thresh = best_params.get('zcr_threshold', best_params.get('zcr_voiced_unvoiced', 0))
    st_thresh = best_params.get('st_threshold', best_params.get('st_voiced_unvoiced', 0.7))
    
    print(f"\\nNGƯỚNG TỐI ỬU TÌM ĐƯỢC:")
    print(f"   Frame: {best_params['frame_length']*1000:.0f}ms/{best_params['frame_shift']*1000:.0f}ms")
    print(f"   Energy Threshold: {energy_thresh:.6f}")
    print(f"   ZCR Threshold: {zcr_thresh:.6f}")  
    print(f"   ST Threshold: {st_thresh:.6f}")
    print(f"   Score: {best_params['score']:.4f}")
    
    print("\\n🎯 THRESHOLD OPTIMIZATION COMPLETED!")
    print("📄 Files saved:")
    print(f"   • {best_params_file}")
    print(f"   • {report_file}")
    print("✅ Optimization completed with ground truth validation!")

if __name__ == "__main__":
    main()
