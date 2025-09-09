#!/usr/bin/env python3
"""
MODULE ĐÁNH GIÁ SUV CLASSIFICATION
Xử lý demo phân loại, tính accuracy và so sánh với ground truth
Chỉ sử dụng các hàm tự viết và built-in functions của Python/Numpy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from audio_analyzer import AudioAnalyzer
from suv_classifier import SUVClassifier
from plotter import plot_comparison_with_ground_truth

def find_test_files_with_labels(test_dir):
    """
    Tìm các file test có cả .wav và .lab để demo với ground truth
    
    Args:
        test_dir: Thư mục chứa file test
        
    Returns:
        List[Tuple]: Danh sách (wav_path, lab_path)
    """
    test_files = []
    
    if not os.path.exists(test_dir):
        return test_files
    
    # Ưu tiên các file cố định
    priority_files = ['phone_F2.wav', 'phone_M2.wav', 'studio_F2.wav', 'studio_M2.wav']
    
    # Tìm file theo thứ tự ưu tiên
    for filename in priority_files:
        wav_path = os.path.join(test_dir, filename)
        lab_path = wav_path.replace('.wav', '.lab')
        
        if os.path.exists(wav_path) and os.path.exists(lab_path):
            test_files.append((wav_path, lab_path))
    
    # Nếu chưa đủ 4 file, tìm thêm file khác
    if len(test_files) < 4:
        all_wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        
        for filename in all_wav_files:
            if filename not in priority_files:  # Bỏ qua file đã có
                wav_path = os.path.join(test_dir, filename)
                lab_path = wav_path.replace('.wav', '.lab')
                
                if os.path.exists(lab_path):
                    test_files.append((wav_path, lab_path))
                    
                if len(test_files) >= 4:
                    break
    
    return test_files[:4]  # Chỉ lấy tối đa 4 file


def demo_classification_with_ground_truth(test_files, thresholds, results_dir):
    """
    Demo phân loại với ngưỡng tối ưu và so sánh với ground truth
    
    Args:
        test_files: Danh sách file test
        thresholds: Ngưỡng tối ưu
        results_dir: Thư mục lưu kết quả
    """
    print("Bắt đầu demo phân loại với ngưỡng tối ưu...")
    
    # Khởi tạo classifier với ngưỡng tối ưu
    classifier = SUVClassifier(
        frame_length=thresholds['frame_length'],
        frame_shift=thresholds['frame_shift'],
        sr=16000
    )
    
    # Set ngưỡng tối ưu
    classifier.set_thresholds(
        ste_threshold=thresholds['ste_threshold'],
        zcr_threshold=thresholds['zcr_threshold'],
        st_threshold=thresholds['st_threshold']
    )
    
    # Khởi tạo evaluator
    analyzer = AudioAnalyzer(
        frame_length=thresholds['frame_length'],
        frame_shift=thresholds['frame_shift'],
        sr=16000
    )
    
    # Xử lý từng file test với ground truth
    for i, (wav_path, lab_path) in enumerate(test_files):
        filename = os.path.basename(wav_path)
        print(f"\nXử lý file {i+1}/{len(test_files)}: {filename}")
        
        try:
            # Phân loại với ngưỡng tối ưu
            audio, ste, zcr, st, predictions = classifier.classify(wav_path)
            
            # Làm mịn dự đoán
            smoothed_predictions = classifier.smooth_predictions(predictions, min_segment_length=30)
            
            # Load ground truth từ file .lab
            print(f"  Sử dụng ground truth: {os.path.basename(lab_path)}")
            
            segments = analyzer.load_labels(lab_path)
            true_labels = analyzer.get_frame_labels(segments, len(audio))
            
            # Đảm bảo chiều dài khớp nhau
            min_length = min(len(true_labels), len(smoothed_predictions))
            true_labels = true_labels[:min_length]
            smoothed_predictions = smoothed_predictions[:min_length]
            
            # Tính độ chính xác
            accuracy = compute_accuracy_manual(true_labels, smoothed_predictions)
            print(f"  Độ chính xác: {accuracy:.4f}")
            
            # Tính accuracy cho từng class
            class_accuracies = compute_class_accuracies(true_labels, smoothed_predictions)
            print(f"  Silence Acc: {class_accuracies[0]:.4f}, Voiced Acc: {class_accuracies[1]:.4f}, Unvoiced Acc: {class_accuracies[2]:.4f}")
            
            # Vẽ và so sánh với ground truth
            plot_comparison_with_ground_truth(
                audio=audio,
                ste=ste,
                zcr=zcr,
                st=st,
                predictions=predictions,
                smoothed_predictions=smoothed_predictions,
                true_labels=true_labels,
                filename=filename,
                thresholds=thresholds,
                figure_position=i,
                results_dir=results_dir
            )
            
        except Exception as e:
            print(f"  ✗ Lỗi xử lý {filename}: {e}")
    
    print("\n🎯 Demo phân loại với ground truth hoàn thành!")
    plt.show()  # Hiển thị tất cả figure


def compute_accuracy_manual(y_true, y_pred):
    """
    Tính accuracy thủ công thay thế sklearn
    
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


def compute_class_accuracies(y_true, y_pred):
    """
    Tính accuracy cho từng class
    
    Args:
        y_true: Nhãn thực
        y_pred: Nhãn dự đoán
        
    Returns:
        List[float]: Accuracy cho từng class [silence, voiced, unvoiced]
    """
    class_accuracies = []
    
    for class_id in [0, 1, 2]:  # silence, voiced, unvoiced
        mask = [true == class_id for true in y_true]
        if sum(mask) > 0:
            correct = sum(1 for i, (true, pred) in enumerate(zip(y_true, y_pred)) 
                         if mask[i] and true == pred)
            accuracy = correct / sum(mask)
        else:
            accuracy = 0.0
        class_accuracies.append(accuracy)
    
    return class_accuracies


def compute_f1_scores(y_true, y_pred):
    """
    Tính F1 score cho từng class
    
    Args:
        y_true: Nhãn thực
        y_pred: Nhãn dự đoán
        
    Returns:
        Dict: F1 scores cho từng class
    """
    classes = [0, 1, 2]  # Silence, Voiced, Unvoiced
    class_names = ['Silence', 'Voiced', 'Unvoiced']
    
    f1_scores = {}
    
    for cls, name in zip(classes, class_names):
        # True Positive: Dự đoán đúng class này
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        
        # False Positive: Dự đoán nhầm là class này
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        
        # False Negative: Bỏ sót class này
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred != cls)
        
        # Precision và Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores[name] = f1
    
    return f1_scores


def evaluate_single_file(classifier, wav_path, lab_path=None):
    """
    Đánh giá một file duy nhất
    
    Args:
        classifier: SUVClassifier instance
        wav_path: Đường dẫn file audio
        lab_path: Đường dẫn file label (optional)
        
    Returns:
        Dict: Kết quả đánh giá
    """
    # Phân loại
    audio, ste, zcr, st, predictions = classifier.classify(wav_path)
    smoothed_predictions = classifier.smooth_predictions(predictions, min_segment_length=30)
    
    result = {
        'filename': os.path.basename(wav_path),
        'total_frames': len(predictions),
        'predictions': predictions,
        'smoothed_predictions': smoothed_predictions,
        'features': {
            'ste': ste,
            'zcr': zcr,
            'st': st
        }
    }
    
    # Nếu có ground truth
    if lab_path and os.path.exists(lab_path):
        analyzer = AudioAnalyzer(
            frame_length=classifier.analyzer.frame_length,
            frame_shift=classifier.analyzer.frame_shift,
            sr=classifier.analyzer.sr
        )
        
        segments = analyzer.load_labels(lab_path)
        true_labels = analyzer.get_frame_labels(segments, len(audio))
        
        # Đảm bảo chiều dài khớp nhau
        min_length = min(len(true_labels), len(smoothed_predictions))
        true_labels = true_labels[:min_length]
        smoothed_predictions = smoothed_predictions[:min_length]
        
        # Tính các metric
        accuracy = compute_accuracy_manual(true_labels, smoothed_predictions)
        class_accuracies = compute_class_accuracies(true_labels, smoothed_predictions)
        f1_scores = compute_f1_scores(true_labels, smoothed_predictions)
        
        result.update({
            'has_ground_truth': True,
            'true_labels': true_labels,
            'accuracy': accuracy,
            'class_accuracies': {
                'silence': class_accuracies[0],
                'voiced': class_accuracies[1],
                'unvoiced': class_accuracies[2]
            },
            'f1_scores': f1_scores
        })
    else:
        result['has_ground_truth'] = False
    
    return result


def save_evaluation_results(results, results_dir):
    """
    Lưu kết quả đánh giá vào file
    
    Args:
        results: List các kết quả đánh giá
        results_dir: Thư mục lưu kết quả
    """
    report_file = os.path.join(results_dir, "evaluation_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== BÁO CÁO ĐÁNH GIÁ SUV CLASSIFICATION ===\n\n")
        
        total_accuracy = 0
        total_files = 0
        
        for result in results:
            f.write(f"FILE: {result['filename']}\n")
            f.write(f"Tổng số frames: {result['total_frames']}\n")
            
            if result['has_ground_truth']:
                f.write(f"Độ chính xác tổng thể: {result['accuracy']:.4f}\n")
                f.write(f"Accuracy theo class:\n")
                f.write(f"  - Silence: {result['class_accuracies']['silence']:.4f}\n")
                f.write(f"  - Voiced: {result['class_accuracies']['voiced']:.4f}\n")
                f.write(f"  - Unvoiced: {result['class_accuracies']['unvoiced']:.4f}\n")
                f.write(f"F1 scores:\n")
                for class_name, f1 in result['f1_scores'].items():
                    f.write(f"  - {class_name}: {f1:.4f}\n")
                
                total_accuracy += result['accuracy']
                total_files += 1
            else:
                f.write("Không có ground truth để đánh giá\n")
            
            f.write("\n" + "-"*50 + "\n")
        
        if total_files > 0:
            f.write(f"\nTÓM TẮT:\n")
            f.write(f"Độ chính xác trung bình: {total_accuracy/total_files:.4f}\n")
            f.write(f"Số file có ground truth: {total_files}\n")
    
    print(f"Đã lưu báo cáo đánh giá: {report_file}")
