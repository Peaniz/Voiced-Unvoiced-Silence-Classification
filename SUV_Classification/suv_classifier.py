import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from audio_analyzer import AudioAnalyzer

class SUVClassifier:
    """
    Lớp phân loại Speech/Unvoiced/Voiced (SUV) sử dụng STE và ZCR
    """
    
    def __init__(self, frame_length=0.025, frame_shift=0.01, sr=16000):
        """
        Khởi tạo SUV Classifier
        
        Args:
            frame_length (float): Độ dài khung (giây)
            frame_shift (float): Bước nhảy khung (giây)
            sr (int): Tần số lấy mẫu
        """
        self.analyzer = AudioAnalyzer(frame_length, frame_shift, sr)
        self.ste_thresholds = {'speech_silence': 0, 'voiced_unvoiced': 0}
        self.zcr_thresholds = {'speech_silence': 0, 'voiced_unvoiced': 0.5}
        self.st_thresholds = {'speech_silence': 0, 'voiced_unvoiced': 0.7}  # Default SUVDA
        self.trained = False
        
    def train(self, training_files: Union[List[str], List[Tuple[str, str]]]) -> Dict:
        """
        Huấn luyện DYNAMIC THRESHOLD theo bài báo (không cần ground truth)
        
        Args:
            training_files: List audio files (wav_path) hoặc (wav_path, lab_path) - chỉ dùng wav
            
        Returns:
            Dict: Thống kê và ngưỡng dynamic tìm được
        """
        print("=== DYNAMIC THRESHOLD TRAINING (UNSUPERVISED) ===")
        print("Theo bài báo: T = (W × M1 + M2) / (W + 1)")
        
        all_ste_values = []
        all_zcr_values = []
        all_st_values = []
        
        # Chuẩn hóa input (chỉ lấy wav paths)
        wav_files = []
        if isinstance(training_files[0], tuple):
            wav_files = [wav_path for wav_path, _ in training_files]
        else:
            wav_files = training_files
        
        print(f"Thu thập features từ {len(wav_files)} file audio...")
        
        for wav_path in wav_files:
            print(f"Xử lý file: {wav_path}")
            
            # Chỉ load audio, không cần ground truth
            audio, _ = self.analyzer.load_audio(wav_path)
            
            # Tính STE, ZCR và ST
            ste = self.analyzer.compute_ste(audio)
            zcr = self.analyzer.compute_zcr(audio)
            st = self.analyzer.compute_spectrum_tilt(audio)
            
            # Thu thập tất cả giá trị (không phân loại theo ground truth)
            all_ste_values.extend(ste)
            all_zcr_values.extend(zcr)
            all_st_values.extend(st)
        
        print(f"Tổng cộng thu thập: {len(all_ste_values)} frames")
        
        # Tính DYNAMIC THRESHOLDS theo công thức T = (W × M1 + M2) / (W + 1)
        dynamic_stats = self._compute_dynamic_thresholds(
            all_ste_values, all_zcr_values, all_st_values
        )
        
        self.trained = True
        
        print(f"\\n=== NGUONG DYNAMIC (UNSUPERVISED) ===")
        print(f"Energy Threshold (STE): {self.ste_thresholds['speech_silence']:.6f}")
        print(f"ZCR Threshold: {self.zcr_thresholds['voiced_unvoiced']:.6f}")
        print(f"Spectrum Tilt Threshold: {self.st_thresholds['voiced_unvoiced']:.6f}")
        
        return {
            'dynamic_stats': dynamic_stats,
            'ste_thresholds': self.ste_thresholds,
            'zcr_thresholds': self.zcr_thresholds,
            'st_thresholds': self.st_thresholds
        }
    
    def _compute_dynamic_thresholds(self, ste_values: List[float], zcr_values: List[float], st_values: List[float]):
        """
        Tính DYNAMIC THRESHOLDS theo công thức bài báo: T = (W × M1 + M2) / (W + 1)
        
        Args:
            ste_values: Tất cả giá trị STE từ audio files
            zcr_values: Tất cả giá trị ZCR từ audio files  
            st_values: Tất cả giá trị ST từ audio files
            
        Returns:
            Dict: Thống kê histogram và threshold dynamics
        """
        print("\\n=== DYNAMIC THRESHOLD COMPUTATION ===")
        
        # Convert to numpy arrays
        ste_array = np.array(ste_values)
        zcr_array = np.array(zcr_values)
        st_array = np.array(st_values)
        
        # Tính dynamic threshold cho từng feature
        ste_threshold, ste_hist_stats = self._dynamic_threshold_single_feature(
            ste_array, "STE (Short-time Energy)", W=1.0
        )
        
        zcr_threshold, zcr_hist_stats = self._dynamic_threshold_single_feature(
            zcr_array, "ZCR (Zero Crossing Rate)", W=0.8
        )
        
        st_threshold, st_hist_stats = self._dynamic_threshold_single_feature(
            st_array, "ST (Spectrum Tilt)", W=1.2
        )
        
        # Set computed thresholds
        self.ste_thresholds['speech_silence'] = ste_threshold
        self.ste_thresholds['voiced_unvoiced'] = 0
        
        self.zcr_thresholds['voiced_unvoiced'] = zcr_threshold  
        self.zcr_thresholds['speech_silence'] = 0
        
        self.st_thresholds['voiced_unvoiced'] = max(0.3, min(0.9, st_threshold))  # Clamp ST
        self.st_thresholds['speech_silence'] = 0
        
        print(f"\\n=== KET QUA DYNAMIC THRESHOLDS ===")
        print(f"T_STE = {ste_threshold:.6f} (Speech/Silence separation)")
        print(f"T_ZCR = {zcr_threshold:.6f} (Voiced/Unvoiced discrimination)")
        print(f"T_ST = {self.st_thresholds['voiced_unvoiced']:.6f} (Voiced/Unvoiced discrimination)")
        
        return {
            'ste': ste_hist_stats,
            'zcr': zcr_hist_stats,
            'st': st_hist_stats,
            'feature_stats': {
                'ste': {'mean': np.mean(ste_array), 'std': np.std(ste_array), 'min': np.min(ste_array), 'max': np.max(ste_array)},
                'zcr': {'mean': np.mean(zcr_array), 'std': np.std(zcr_array), 'min': np.min(zcr_array), 'max': np.max(zcr_array)},
                'st': {'mean': np.mean(st_array), 'std': np.std(st_array), 'min': np.min(st_array), 'max': np.max(st_array)}
            }
        }
    
    def _dynamic_threshold_single_feature(self, feature_values: np.ndarray, feature_name: str, W: float = 1.0):
        """
        Tính dynamic threshold cho một feature theo công thức: T = (W × M1 + M2) / (W + 1)
        
        Args:
            feature_values: Array giá trị của feature
            feature_name: Tên feature để debug
            W: User-defined parameter (default=1.0)
            
        Returns:
            Tuple: (threshold, histogram_stats)
        """
        try:
            print(f"      Computing {feature_name} threshold (W={W})")
            
            # VALIDATION
            if len(feature_values) < 50:
                raise ValueError(f"Not enough data: {len(feature_values)} samples")
                
            if np.isnan(feature_values).any() or np.isinf(feature_values).any():
                print(f"      Warning: Invalid values in {feature_name}, cleaning...")
                feature_values = feature_values[np.isfinite(feature_values)]
                
            if len(feature_values) < 50:
                raise ValueError(f"Not enough valid data after cleaning")
            
            # QUICK STATS
            f_min, f_max = np.min(feature_values), np.max(feature_values)
            f_mean, f_std = np.mean(feature_values), np.std(feature_values)
            print(f"      Range: [{f_min:.4f}, {f_max:.4f}], μ±σ: {f_mean:.4f}±{f_std:.4f}")
            
            # Bước 1: Tính histogram (REDUCED BINS)
            n_bins = min(30, max(10, len(feature_values)//100))  # Adaptive bins
            hist, bin_edges = np.histogram(feature_values, bins=n_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # VALIDATE HISTOGRAM
            if np.sum(hist) == 0:
                raise ValueError("Empty histogram")
            
            # Bước 2: Tìm local maxima với TIMEOUT PROTECTION
            from scipy.ndimage import gaussian_filter1d
            smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=0.5)  # Lighter smoothing
            
            # Tìm peaks với minimum height và distance
            min_height = np.max(smoothed_hist) * 0.05  # LOWER threshold (5%)
            peaks, _ = find_peaks(smoothed_hist, height=min_height, distance=2)  # SHORTER distance
            
            if len(peaks) < 2:
                print(f"      Only {len(peaks)} peaks found, using fallback")
                # Fallback: Dùng percentiles
                if "STE" in feature_name:
                    threshold = np.percentile(feature_values, 25)
                elif "ZCR" in feature_name:
                    threshold = np.median(feature_values)
                elif "ST" in feature_name:
                    threshold = np.percentile(feature_values, 60)
                else:
                    threshold = np.median(feature_values)
            else:
                # Sắp xếp peaks theo height (descending) - SAFE OPERATION
                peak_heights = smoothed_hist[peaks]
                if len(peak_heights) > 0:
                    sorted_indices = np.argsort(peak_heights)[::-1]
                    sorted_peaks = peaks[sorted_indices]
                    
                    # Lấy 2 peaks cao nhất
                    M1_idx = sorted_peaks[0]
                    M2_idx = sorted_peaks[1] if len(sorted_peaks) > 1 else sorted_peaks[0]
                    
                    M1 = bin_centers[M1_idx]
                    M2 = bin_centers[M2_idx]
                    
                    # SAFE COMPUTATION - Áp dụng công thức T = (W × M1 + M2) / (W + 1)
                    if W > 0:
                        threshold = (W * M1 + M2) / (W + 1)
                    else:
                        threshold = (M1 + M2) / 2  # Fallback for W=0
                    
                    print(f"      M1={M1:.4f}, M2={M2:.4f} → T={threshold:.6f}")
                else:
                    threshold = np.median(feature_values)
            
            # VALIDATE THRESHOLD
            if np.isnan(threshold) or np.isinf(threshold):
                print(f"      Invalid threshold, using median")
                threshold = np.median(feature_values)
            
            # SIMPLE STATS (no complex objects to avoid memory issues)
            hist_stats = {
                'threshold': threshold,
                'W': W,
                'n_peaks': len(peaks) if 'peaks' in locals() else 0,
                'feature_mean': f_mean,
                'feature_std': f_std
            }
            
            return threshold, hist_stats
            
        except Exception as e:
            print(f"      ERROR in {feature_name} threshold computation: {str(e)}")
            # ULTIMATE FALLBACK
            if "STE" in feature_name:
                fallback = np.percentile(feature_values, 25)
            elif "ZCR" in feature_name:
                fallback = np.median(feature_values) 
            elif "ST" in feature_name:
                fallback = np.percentile(feature_values, 60)
            else:
                fallback = np.median(feature_values)
            
            return fallback, {'threshold': fallback, 'W': W, 'error': str(e)}
    
    def classify(self, wav_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Phân loại SUV cho một file audio sử dụng STE + ZCR + ST (SUVDA)
        
        Args:
            wav_path (str): Đường dẫn file audio
            
        Returns:
            Tuple: (audio, ste, zcr, st, predictions)
        """
        if not self.trained:
            raise ValueError("Classifier chưa được huấn luyện!")
        
        # Load audio
        audio, _ = self.analyzer.load_audio(wav_path)
        
        # Tính STE, ZCR và ST
        ste = self.analyzer.compute_ste(audio)
        zcr = self.analyzer.compute_zcr(audio)
        st = self.analyzer.compute_spectrum_tilt(audio)
        
        # Đảm bảo chiều dài khớp nhau
        min_length = min(len(ste), len(zcr), len(st))
        ste = ste[:min_length]
        zcr = zcr[:min_length]
        st = st[:min_length]
        
        # Logic phân loại SUVDA với STE + ZCR + ST
        predictions = np.zeros(min_length, dtype=int)
        
        # Lấy 3 ngưỡng chính với fallback
        energy_threshold = self.ste_thresholds.get('speech_silence', 0)
        zcr_threshold = self.zcr_thresholds.get('voiced_unvoiced', 0.5)  
        st_threshold = self.st_thresholds.get('voiced_unvoiced', 0.7)  # Default từ bài báo SUVDA
        
        for i in range(min_length):
            # Logic theo bài báo SUVDA:
            # - Silence: ZCR < T_ZCR, STE < T_STE
            # - Unvoiced: ZCR > T_ZCR, STE > T_STE, ST < 0.7
            # - Voiced: STE > T_STE, ST > 0.7, ZCR thấp
            
            # BƯỚC 1: Kiểm tra SILENCE
            if ste[i] < energy_threshold and zcr[i] < zcr_threshold:
                # Cả STE và ZCR đều thấp → SILENCE
                predictions[i] = 0
            elif ste[i] < energy_threshold:
                # Chỉ STE thấp nhưng ZCR cao → vẫn có thể là silence hoặc unvoiced nhỏ
                predictions[i] = 0
            else:
                # STE cao → đây là SPEECH
                # BƯỚC 2: Phân biệt VOICED vs UNVOICED bằng ST + ZCR
                
                if st[i] > st_threshold and zcr[i] < zcr_threshold:
                    # ST cao (năng lượng ở tần số thấp) + ZCR thấp → VOICED
                    predictions[i] = 1
                elif st[i] < st_threshold and zcr[i] > zcr_threshold:  
                    # ST thấp (năng lượng ở tần số cao) + ZCR cao → UNVOICED
                    predictions[i] = 2
                else:
                    # Trường hợp không rõ ràng, quyết định dựa trên ST chủ yếu
                    if st[i] > st_threshold:
                        predictions[i] = 1  # VOICED
                    else:
                        predictions[i] = 2  # UNVOICED
        
        return audio, ste, zcr, st, predictions
    
    def smooth_predictions(self, predictions: np.ndarray, min_segment_length: int = 30) -> np.ndarray:
        """
        Làm mịn dự đoán với nhiều bước xử lý
        
        Args:
            predictions (np.ndarray): Dự đoán gốc
            min_segment_length (int): Độ dài tối thiểu của segment (số frame)
            
        Returns:
            np.ndarray: Dự đoán đã được làm mịn
        """
        smoothed = predictions.copy()
        
        # Bước 1: Median filter để loại bỏ noise
        from scipy import signal
        if len(smoothed) > 5:
            smoothed = signal.medfilt(smoothed, kernel_size=5)
            smoothed = smoothed.astype(int)
        
        # Bước 2: Loại bỏ các đoạn quá ngắn cho tất cả các class
        for target_class in [0, 1, 2]:  # silence, voiced, unvoiced
            i = 0
            while i < len(smoothed):
                if smoothed[i] == target_class:
                    # Tìm độ dài đoạn
                    j = i
                    while j < len(smoothed) and smoothed[j] == target_class:
                        j += 1
                    
                    # Nếu đoạn quá ngắn, thay thế bằng class xung quanh
                    segment_length = j - i
                    min_length = min_segment_length if target_class == 0 else min_segment_length // 3  # Yêu cầu ngắn hơn cho speech
                    
                    if segment_length < min_length:
                        # Tìm class thay thế tốt nhất
                        before_label = smoothed[i-1] if i > 0 else -1
                        after_label = smoothed[j] if j < len(smoothed) else -1
                        
                        # Chọn replacement label
                        if before_label == after_label and before_label != -1:
                            # Cùng class ở 2 bên
                            replacement = before_label
                        elif before_label != -1 and after_label != -1:
                            # Khác class, chọn theo priority: voiced > unvoiced > silence
                            candidates = [before_label, after_label]
                            if 1 in candidates:  # voiced
                                replacement = 1
                            elif 2 in candidates:  # unvoiced  
                                replacement = 2
                            else:  # silence
                                replacement = 0
                        elif before_label != -1:
                            replacement = before_label
                        elif after_label != -1:
                            replacement = after_label
                        else:
                            replacement = 1 if target_class == 0 else 0  # Default replacement
                        
                        smoothed[i:j] = replacement
                    
                    i = j
                else:
                    i += 1
        
        # Bước 3: Majority voting trong sliding window
        window_size = 7
        if len(smoothed) > window_size:
            final_smoothed = smoothed.copy()
            for i in range(len(smoothed)):
                start = max(0, i - window_size//2)
                end = min(len(smoothed), i + window_size//2 + 1)
                window = smoothed[start:end]
                
                # Tìm class xuất hiện nhiều nhất trong window
                unique, counts = np.unique(window, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                final_smoothed[i] = majority_class
            
            smoothed = final_smoothed
        
        return smoothed
