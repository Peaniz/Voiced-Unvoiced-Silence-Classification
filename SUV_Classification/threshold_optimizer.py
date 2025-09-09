import numpy as np
from typing import Dict, List, Tuple, Optional
import itertools
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from suv_classifier import SUVClassifier
from audio_analyzer import AudioAnalyzer
from evaluator import SUVEvaluator
import warnings
warnings.filterwarnings('ignore')

class ThresholdOptimizer:
    """
    Tự động tìm ngưỡng tối ưu cho SUV classification
    """
    
    def __init__(self):
        self.best_params = {}
        self.best_score = 0
        self.optimization_history = []
        
    def optimize_thresholds_grid_search(self, training_files: List[Tuple[str, str]], 
                                      validation_split: float = 0.5,
                                      verbose: bool = True) -> Dict:
        """
        Tối ưu ngưỡng bằng grid search với cross-validation
        
        Args:
            training_files: List file (wav, lab)
            validation_split: Tỷ lệ chia validation
            verbose: In log hay không
            
        Returns:
            Dict: Best parameters tìm được
        """
        print("=== TỐI ỬU NGƯỠNG TỰ ĐỘNG BẰNG GRID SEARCH ===\\n")
        
        # Chia train/validation
        n_train = int(len(training_files) * (1 - validation_split))
        train_files = training_files[:n_train] 
        val_files = training_files[n_train:]
        
        if len(val_files) == 0:
            val_files = train_files  # Fallback nếu ít file
            
        print(f"Training files: {len(train_files)}")
        print(f"Validation files: {len(val_files)}\\n")
        
        # Định nghĩa search space
        frame_lengths = [0.020, 0.025, 0.030]  # 20ms, 25ms, 30ms
        frame_shifts = [0.008, 0.010, 0.012]   # 8ms, 10ms, 12ms  
        
        # Tối ưu 3 ngưỡng chính theo SUVDA
        # Energy threshold factors (STE cho silence vs speech)
        energy_threshold_factors = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
        
        # ZCR threshold factors (ZCR cho voiced vs unvoiced)
        zcr_threshold_factors = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5]
        
        # ST threshold factors (Spectrum Tilt cho voiced vs unvoiced)
        st_threshold_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]  # Xung quanh 0.7 từ bài báo
        
        best_score = 0
        best_params = {}
        iteration = 0
        total_iterations = (len(frame_lengths) * len(frame_shifts) * 
                          len(energy_threshold_factors) * len(zcr_threshold_factors) * 
                          len(st_threshold_factors))
        
        print(f"Tổng số combinations cần test: {total_iterations}\\n")
        
        # Grid search
        for frame_length in frame_lengths:
            for frame_shift in frame_shifts:
                # Train classifier với frame parameters này
                classifier = SUVClassifier(frame_length=frame_length, 
                                         frame_shift=frame_shift, 
                                         sr=16000)
                
                try:
                    # Huấn luyện để lấy baseline statistics
                    baseline_stats = classifier.train(train_files)
                    
                    # Lấy baseline thresholds (3 ngưỡng chính)
                    baseline_energy_threshold = baseline_stats['ste_thresholds']['speech_silence']
                    baseline_zcr_threshold = baseline_stats['zcr_thresholds']['voiced_unvoiced']
                    baseline_st_threshold = baseline_stats['st_thresholds']['voiced_unvoiced']
                    
                    # Test different threshold combinations (3 vòng lặp)
                    for energy_factor in energy_threshold_factors:
                        for zcr_factor in zcr_threshold_factors:
                            for st_factor in st_threshold_factors:
                                iteration += 1
                                
                                if iteration % 50 == 0:
                                    print(f"Progress: {iteration}/{total_iterations} ({iteration/total_iterations*100:.1f}%)")
                                
                                # Set custom thresholds (3 ngưỡng chính)
                                energy_threshold = baseline_energy_threshold * energy_factor
                                zcr_threshold = baseline_zcr_threshold * zcr_factor
                                st_threshold = baseline_st_threshold * st_factor
                                
                                # Clamp ST threshold vào khoảng hợp lý
                                st_threshold = max(0.3, min(0.95, st_threshold))
                                
                                classifier.ste_thresholds['speech_silence'] = energy_threshold
                                classifier.ste_thresholds['voiced_unvoiced'] = 0  # Không dùng
                                classifier.zcr_thresholds['speech_silence'] = 0   # Không dùng
                                classifier.zcr_thresholds['voiced_unvoiced'] = zcr_threshold
                                classifier.st_thresholds['speech_silence'] = 0    # Không dùng
                                classifier.st_thresholds['voiced_unvoiced'] = st_threshold
                                
                                # Đánh giá trên validation set
                                score = self._evaluate_params(classifier, val_files)
                                
                                # Track history
                                params = {
                                    'frame_length': frame_length,
                                    'frame_shift': frame_shift,
                                    'energy_threshold': energy_threshold,
                                    'zcr_threshold': zcr_threshold,
                                    'st_threshold': st_threshold,
                                    # Để tương thích với code cũ
                                    'ste_speech_silence': energy_threshold,
                                    'ste_voiced_unvoiced': 0,
                                    'zcr_speech_silence': 0,
                                    'zcr_voiced_unvoiced': zcr_threshold,
                                    'st_speech_silence': 0,
                                    'st_voiced_unvoiced': st_threshold,
                                    'score': score
                                }
                                
                                self.optimization_history.append(params)
                                
                                # Update best
                                if score > best_score:
                                    best_score = score
                                    best_params = params.copy()
                                    
                                    if verbose:
                                        print(f"\\n🎯 NEW BEST SCORE: {score:.4f}")
                                        print(f"   Frame: {frame_length*1000:.0f}ms/{frame_shift*1000:.0f}ms")
                                        print(f"   Energy Threshold: {energy_threshold:.6f}")
                                        print(f"   ZCR Threshold: {zcr_threshold:.6f}")
                                        print(f"   🆕 ST Threshold: {st_threshold:.6f}")
                                
                except Exception as e:
                    if verbose:
                        print(f"   Error with frame_length={frame_length}, frame_shift={frame_shift}: {e}")
                    continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        print(f"\\n=== KẾT QUẢ TỐI ỬU ===")
        print(f"Best Score: {best_score:.4f}")
        print(f"Best Frame Length: {best_params['frame_length']*1000:.0f}ms")
        print(f"Best Frame Shift: {best_params['frame_shift']*1000:.0f}ms")
        print(f"🎯 Best Energy Threshold: {best_params['energy_threshold']:.6f}")
        print(f"🎯 Best ZCR Threshold: {best_params['zcr_threshold']:.6f}")
        print(f"🆕 Best ST Threshold: {best_params['st_threshold']:.6f}")
        
        print(f"\\n📋 Chi tiết 3 ngưỡng chính (SUVDA):")
        print(f"   1. Energy Threshold (STE): Tách silence vs speech = {best_params['energy_threshold']:.6f}")
        print(f"   2. ZCR Threshold: Tách voiced vs unvoiced = {best_params['zcr_threshold']:.6f}")
        print(f"   3. 🆕 Spectrum Tilt Threshold: Voiced>ST vs Unvoiced<ST = {best_params['st_threshold']:.6f}")
        
        return best_params
    
    def _evaluate_params(self, classifier: SUVClassifier, val_files: List[Tuple[str, str]]) -> float:
        """
        Đánh giá một bộ parameters trên validation set
        
        Args:
            classifier: Classifier đã được config
            val_files: Validation files
            
        Returns:
            float: Điểm đánh giá (F1-weighted)
        """
        analyzer = AudioAnalyzer(classifier.analyzer.frame_length, 
                               classifier.analyzer.frame_shift, 
                               classifier.analyzer.sr)
        
        all_true_labels = []
        all_pred_labels = []
        
        try:
            for wav_path, lab_path in val_files:
                # Classify (giờ có thêm ST)
                audio, ste, zcr, st, predictions = classifier.classify(wav_path)
                smoothed_predictions = classifier.smooth_predictions(predictions, min_segment_length=30)
                
                # Ground truth
                segments = analyzer.load_labels(lab_path)
                true_labels = analyzer.get_frame_labels(segments, len(audio))
                
                # Align lengths
                min_length = min(len(true_labels), len(smoothed_predictions))
                true_labels = true_labels[:min_length]
                smoothed_predictions = smoothed_predictions[:min_length]
                
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(smoothed_predictions)
            
            # Tính weighted F1 score
            if len(all_true_labels) > 0:
                f1_weighted = f1_score(all_true_labels, all_pred_labels, average='weighted')
                accuracy = accuracy_score(all_true_labels, all_pred_labels)
                
                # Combined score: 70% F1 + 30% accuracy
                combined_score = 0.7 * f1_weighted + 0.3 * accuracy
                return combined_score
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def adaptive_threshold_per_file(self, classifier: SUVClassifier, 
                                  training_files: List[Tuple[str, str]],
                                  target_file: str) -> Dict:
        """
        Tìm ngưỡng adaptive cho từng file cụ thể
        
        Args:
            classifier: Base classifier
            training_files: Training data
            target_file: File cần tối ưu ngưỡng
            
        Returns:
            Dict: Adaptive thresholds
        """
        print(f"\\n=== ADAPTIVE THRESHOLDING CHO {target_file} ===")
        
        # Load target file để phân tích
        analyzer = AudioAnalyzer(classifier.analyzer.frame_length,
                               classifier.analyzer.frame_shift,
                               classifier.analyzer.sr)
        
        audio, _ = analyzer.load_audio(target_file)
        ste = analyzer.compute_ste(audio)
        zcr = analyzer.compute_zcr(audio)
        
        # Phân tích đặc tính của file này
        ste_stats = {
            'mean': np.mean(ste),
            'std': np.std(ste),
            'min': np.min(ste),
            'max': np.max(ste),
            'median': np.median(ste)
        }
        
        zcr_stats = {
            'mean': np.mean(zcr),
            'std': np.std(zcr),
            'min': np.min(zcr),
            'max': np.max(zcr),
            'median': np.median(zcr)
        }
        
        # Tính ST cho file này
        st = analyzer.compute_spectrum_tilt(audio)
        st_stats = {
            'mean': np.mean(st),
            'std': np.std(st),
            'min': np.min(st),
            'max': np.max(st),
            'median': np.median(st)
        }
        
        # Adaptive thresholds - 3 ngưỡng chính
        adaptive_thresholds = {
            # Energy threshold dựa trên percentile của STE
            'energy_threshold': np.percentile(ste, 20),  # 20th percentile tách silence vs speech
            'zcr_threshold': zcr_stats['median'],  # Median ZCR tách voiced vs unvoiced
            'st_threshold': max(0.5, min(0.85, st_stats['mean'])),  # ST threshold adaptive
            
            # Để tương thích với code cũ
            'ste_speech_silence': np.percentile(ste, 20),
            'ste_voiced_unvoiced': 0,
            'zcr_speech_silence': 0,
            'zcr_voiced_unvoiced': zcr_stats['median'],
            'st_speech_silence': 0,
            'st_voiced_unvoiced': max(0.5, min(0.85, st_stats['mean']))
        }
        
        print(f"File characteristics:")
        print(f"  STE: mean={ste_stats['mean']:.4f}, std={ste_stats['std']:.4f}")
        print(f"  ZCR: mean={zcr_stats['mean']:.4f}, std={zcr_stats['std']:.4f}")
        print(f"  🆕 ST: mean={st_stats['mean']:.4f}, std={st_stats['std']:.4f}")
        
        print(f"🎯 Adaptive thresholds (3 ngưỡng SUVDA):")
        print(f"  Energy Threshold: {adaptive_thresholds['energy_threshold']:.6f}")
        print(f"  ZCR Threshold: {adaptive_thresholds['zcr_threshold']:.6f}")
        print(f"  🆕 ST Threshold: {adaptive_thresholds['st_threshold']:.6f}")
        
        print(f"\\n📋 Logic SUVDA áp dụng:")
        print(f"  1. Silence: STE < {adaptive_thresholds['energy_threshold']:.4f} AND ZCR < {adaptive_thresholds['zcr_threshold']:.4f}")
        print(f"  2. Voiced: STE ≥ {adaptive_thresholds['energy_threshold']:.4f} AND ST > {adaptive_thresholds['st_threshold']:.4f} AND ZCR < {adaptive_thresholds['zcr_threshold']:.4f}")
        print(f"  3. Unvoiced: STE ≥ {adaptive_thresholds['energy_threshold']:.4f} AND ST < {adaptive_thresholds['st_threshold']:.4f} AND ZCR > {adaptive_thresholds['zcr_threshold']:.4f}")
        
        return adaptive_thresholds
    
    def get_optimization_report(self) -> str:
        """
        Tạo báo cáo chi tiết về quá trình optimization
        """
        if not self.optimization_history:
            return "Chưa có dữ liệu optimization"
        
        history = self.optimization_history
        
        report = "=== BÁO CÁO TỐI ỬU NGƯỠNG ===\\n\\n"
        
        # Top 10 best configurations
        sorted_history = sorted(history, key=lambda x: x['score'], reverse=True)
        top_10 = sorted_history[:10]
        
        report += "TOP 10 CẤU HÌNH TỐT NHẤT:\\n"
        report += f"{'Rank':<5} {'Score':<8} {'Frame(ms)':<10} {'Energy_Th':<10} {'ZCR_Th':<10} {'ST_Th':<10}\\n"
        report += "-" * 75 + "\\n"
        
        for i, config in enumerate(top_10):
            frame_str = f"{config['frame_length']*1000:.0f}/{config['frame_shift']*1000:.0f}"
            energy_thresh = config.get('energy_threshold', config.get('ste_speech_silence', 0))
            zcr_thresh = config.get('zcr_threshold', config.get('zcr_voiced_unvoiced', 0))
            st_thresh = config.get('st_threshold', config.get('st_voiced_unvoiced', 0.7))
            report += f"{i+1:<5} {config['score']:<8.4f} {frame_str:<10} "
            report += f"{energy_thresh:<10.4f} {zcr_thresh:<10.4f} {st_thresh:<10.4f}\\n"
        
        # Statistics
        scores = [h['score'] for h in history]
        report += f"\\nTHỐNG KÊ:\\n"
        report += f"Tổng số cấu hình test: {len(history)}\\n"
        report += f"Score cao nhất: {max(scores):.4f}\\n"
        report += f"Score thấp nhất: {min(scores):.4f}\\n"
        report += f"Score trung bình: {np.mean(scores):.4f}\\n"
        report += f"Độ lệch chuẩn: {np.std(scores):.4f}\\n"
        
        return report
    
    def optimize_dynamic_thresholds(self, training_files: List[str], 
                                  verbose: bool = True) -> Dict:
        """
        Tối ưu DYNAMIC THRESHOLDS (không cần ground truth)
        
        Args:
            training_files: List audio files (chỉ .wav, không cần .lab)
            verbose: In log hay không
            
        Returns:
            Dict: Best W parameters tìm được
        """
        print("=== DYNAMIC THRESHOLD OPTIMIZATION (UNSUPERVISED) ===\\n")
        print("Theo bài báo: T = (W × M1 + M2) / (W + 1)")
        
        # Chuẩn hóa input để chỉ lấy wav files
        wav_files = []
        if len(training_files) > 0 and isinstance(training_files[0], tuple):
            wav_files = [wav_path for wav_path, _ in training_files]
        else:
            wav_files = training_files
            
        print(f"Audio files: {len(wav_files)}")
        
        # Search space cho W parameters (GIẢM ĐỂ TRÁNH LOOP)
        W_values = [0.8, 1.0, 1.2]  # Giảm từ 5 xuống 3 values
        frame_lengths = [0.025]  # Chỉ dùng 1 frame length
        frame_shifts = [0.010]   # Chỉ dùng 1 frame shift
        
        total_combinations = len(W_values)**3 * len(frame_lengths) * len(frame_shifts)
        print(f"Search space: {total_combinations} combinations (W_STE × W_ZCR × W_ST × frames)")
        print("CAUTION: Reduced search space to prevent infinite loops")
        
        best_params = None
        best_separation_score = -1
        current_iteration = 0
        max_iterations = 50  # SAFETY LIMIT
        
        for frame_length in frame_lengths:
            for frame_shift in frame_shifts:
                for w_ste in W_values:
                    for w_zcr in W_values:
                        for w_st in W_values:
                            current_iteration += 1
                            
                            # SAFETY CHECK
                            if current_iteration > max_iterations:
                                print(f"SAFETY STOP: Reached max iterations ({max_iterations})")
                                break
                            
                            try:
                                print(f"\\nTesting combination {current_iteration}/{total_combinations}")
                                print(f"  Frame: {frame_length*1000:.0f}ms/{frame_shift*1000:.0f}ms")
                                print(f"  W values: STE={w_ste}, ZCR={w_zcr}, ST={w_st}")
                                
                                # Test combination
                                classifier = SUVClassifier(
                                    frame_length=frame_length,
                                    frame_shift=frame_shift,
                                    sr=16000
                                )
                                
                                # Custom training with specific W values
                                dynamic_stats = self._train_with_custom_W(
                                    classifier, wav_files, w_ste, w_zcr, w_st
                                )
                                
                                # Đánh giá separation quality (unsupervised metric)
                                separation_score = self._evaluate_cluster_separation(dynamic_stats)
                                print(f"  → Separation score: {separation_score:.4f}")
                                
                                params = {
                                    'frame_length': frame_length,
                                    'frame_shift': frame_shift,
                                    'W_STE': w_ste,
                                    'W_ZCR': w_zcr,
                                    'W_ST': w_st,
                                    'separation_score': separation_score,
                                    'energy_threshold': classifier.ste_thresholds['speech_silence'],
                                    'zcr_threshold': classifier.zcr_thresholds['voiced_unvoiced'],
                                    'st_threshold': classifier.st_thresholds['voiced_unvoiced'],
                                    'ste_speech_silence': classifier.ste_thresholds['speech_silence'],
                                    'ste_voiced_unvoiced': 0,
                                    'zcr_speech_silence': 0,
                                    'zcr_voiced_unvoiced': classifier.zcr_thresholds['voiced_unvoiced'],
                                    'st_speech_silence': 0,
                                    'st_voiced_unvoiced': classifier.st_thresholds['voiced_unvoiced'],
                                    'score': separation_score
                                }
                                
                                if separation_score > best_separation_score:
                                    best_separation_score = separation_score
                                    best_params = params.copy()
                                    print(f"  ✅ NEW BEST: {separation_score:.4f}")
                                    
                            except Exception as e:
                                print(f"  ❌ ERROR in iteration {current_iteration}: {str(e)}")
                                continue
                        
                        # NESTED LOOP SAFETY CHECK
                        if current_iteration > max_iterations:
                            break
                    if current_iteration > max_iterations:
                        break
                if current_iteration > max_iterations:
                    break
            if current_iteration > max_iterations:
                break
        
        # FALLBACK NẾUS KHÔNG TÌM ĐƯỢC BEST PARAMS
        if best_params is None:
            print("\\n⚠️  WARNING: No valid parameters found, using fallback!")
            # Train với default W values
            fallback_classifier = SUVClassifier(frame_length=0.025, frame_shift=0.010, sr=16000)
            try:
                fallback_stats = self._train_with_custom_W(fallback_classifier, wav_files, 1.0, 1.0, 1.0)
                fallback_score = self._evaluate_cluster_separation(fallback_stats)
                best_params = {
                    'frame_length': 0.025,
                    'frame_shift': 0.010,
                    'W_STE': 1.0,
                    'W_ZCR': 1.0,
                    'W_ST': 1.0,
                    'separation_score': fallback_score,
                    'energy_threshold': fallback_classifier.ste_thresholds['speech_silence'],
                    'zcr_threshold': fallback_classifier.zcr_thresholds['voiced_unvoiced'],
                    'st_threshold': fallback_classifier.st_thresholds['voiced_unvoiced'],
                    'ste_speech_silence': fallback_classifier.ste_thresholds['speech_silence'],
                    'ste_voiced_unvoiced': 0,
                    'zcr_speech_silence': 0,
                    'zcr_voiced_unvoiced': fallback_classifier.zcr_thresholds['voiced_unvoiced'],
                    'st_speech_silence': 0,
                    'st_voiced_unvoiced': fallback_classifier.st_thresholds['voiced_unvoiced'],
                    'score': fallback_score
                }
                best_separation_score = fallback_score
            except Exception as e:
                print(f"Even fallback failed: {e}")
                return None
        
        self.best_params = best_params
        self.best_score = best_separation_score
        
        print(f"\\n=== DYNAMIC OPTIMIZATION COMPLETED ===")
        print(f"Best Separation Score: {best_separation_score:.4f}")
        print(f"Best Parameters:")
        print(f"  Frame: {best_params['frame_length']*1000:.0f}ms / {best_params['frame_shift']*1000:.0f}ms")
        print(f"  W_STE: {best_params['W_STE']}")
        print(f"  W_ZCR: {best_params['W_ZCR']}")
        print(f"  W_ST: {best_params['W_ST']}")
        print(f"  Dynamic Thresholds:")
        print(f"    T_STE = {best_params['energy_threshold']:.6f}")
        print(f"    T_ZCR = {best_params['zcr_threshold']:.6f}")
        print(f"    T_ST = {best_params['st_threshold']:.6f}")
        
        return best_params
    
    def _train_with_custom_W(self, classifier: SUVClassifier, wav_files: List[str], 
                           w_ste: float, w_zcr: float, w_st: float) -> Dict:
        """
        Train classifier với custom W parameters cho dynamic threshold
        
        Args:
            classifier: SUVClassifier instance
            wav_files: List of audio file paths  
            w_ste, w_zcr, w_st: W parameters cho từng feature
            
        Returns:
            Dict: Dynamic statistics
        """
        # Thu thập features từ tất cả file
        all_ste_values = []
        all_zcr_values = []
        all_st_values = []
        
        max_files = min(len(wav_files), 4)  # GIỚI HẠN SỐ FILE ĐỂ TRÁNH LOOP
        processed_files = 0
        
        for i, wav_path in enumerate(wav_files[:max_files]):
            try:
                print(f"    Processing {i+1}/{max_files}: {wav_path}")
                audio, _ = classifier.analyzer.load_audio(wav_path)
                
                # KIỂM TRA AUDIO VALIDITY
                if len(audio) < 1000:
                    print(f"    Skipping {wav_path}: too short")
                    continue
                
                ste = classifier.analyzer.compute_ste(audio)
                zcr = classifier.analyzer.compute_zcr(audio)
                st = classifier.analyzer.compute_spectrum_tilt(audio)
                
                # KIỂM TRA FEATURES VALIDITY
                if len(ste) > 0 and len(zcr) > 0 and len(st) > 0:
                    all_ste_values.extend(ste)
                    all_zcr_values.extend(zcr)
                    all_st_values.extend(st)
                    processed_files += 1
                    
            except Exception as e:
                print(f"    Error processing {wav_path}: {str(e)}")
                continue
        
        print(f"    Successfully processed {processed_files} files")
        
        # KIỂM TRA ĐỦ DỮ LIỆU
        if len(all_ste_values) < 100:
            raise ValueError(f"Not enough features: only {len(all_ste_values)} frames")
        
        # Tính dynamic thresholds với custom W
        ste_array = np.array(all_ste_values)
        zcr_array = np.array(all_zcr_values)
        st_array = np.array(all_st_values)
        
        # TIMEOUT PROTECTION cho threshold computation
        try:
            ste_threshold, _ = classifier._dynamic_threshold_single_feature(ste_array, "STE", w_ste)
            zcr_threshold, _ = classifier._dynamic_threshold_single_feature(zcr_array, "ZCR", w_zcr)
            st_threshold, _ = classifier._dynamic_threshold_single_feature(st_array, "ST", w_st)
        except Exception as e:
            print(f"    Threshold computation failed: {e}")
            # FALLBACK THRESHOLDS
            ste_threshold = np.percentile(ste_array, 25)
            zcr_threshold = np.median(zcr_array)
            st_threshold = np.percentile(st_array, 60)
        
        # Set thresholds
        classifier.ste_thresholds['speech_silence'] = ste_threshold
        classifier.zcr_thresholds['voiced_unvoiced'] = zcr_threshold
        classifier.st_thresholds['voiced_unvoiced'] = max(0.3, min(0.9, st_threshold))
        classifier.trained = True
        
        return {
            'ste_stats': {'mean': np.mean(ste_array), 'std': np.std(ste_array)},
            'zcr_stats': {'mean': np.mean(zcr_array), 'std': np.std(zcr_array)},
            'st_stats': {'mean': np.mean(st_array), 'std': np.std(st_array)},
            'feature_ranges': {
                'ste_range': np.max(ste_array) - np.min(ste_array),
                'zcr_range': np.max(zcr_array) - np.min(zcr_array), 
                'st_range': np.max(st_array) - np.min(st_array)
            }
        }
    
    def _evaluate_cluster_separation(self, dynamic_stats: Dict) -> float:
        """
        Đánh giá chất lượng clustering/separation (unsupervised metric)
        
        Args:
            dynamic_stats: Statistics từ dynamic threshold computation
            
        Returns:
            float: Separation score (cao hơn = tốt hơn)
        """
        # Sử dụng coefficient of variation và range để đánh giá separation
        ste_stats = dynamic_stats['ste_stats']
        zcr_stats = dynamic_stats['zcr_stats']
        st_stats = dynamic_stats['st_stats']
        ranges = dynamic_stats['feature_ranges']
        
        # Coefficient of variation (std/mean) - cao hơn = phân tán tốt hơn
        ste_cv = ste_stats['std'] / (abs(ste_stats['mean']) + 1e-6)
        zcr_cv = zcr_stats['std'] / (abs(zcr_stats['mean']) + 1e-6)
        st_cv = st_stats['std'] / (abs(st_stats['mean']) + 1e-6)
        
        # Normalized ranges - cao hơn = dynamic range tốt hơn
        ste_norm_range = ranges['ste_range'] / (abs(ste_stats['mean']) + 1e-6)
        zcr_norm_range = ranges['zcr_range'] / (abs(zcr_stats['mean']) + 1e-6)
        st_norm_range = ranges['st_range'] / (abs(st_stats['mean']) + 1e-6)
        
        # Combined separation score
        cv_score = (ste_cv + zcr_cv + st_cv) / 3
        range_score = (ste_norm_range + zcr_norm_range + st_norm_range) / 3
        
        # Weighted combination
        separation_score = 0.6 * cv_score + 0.4 * range_score
        
        return separation_score
