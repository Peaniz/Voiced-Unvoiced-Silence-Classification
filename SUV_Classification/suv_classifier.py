import numpy as np
from typing import Dict, List, Tuple
from audio_analyzer import AudioAnalyzer

class SUVClassifier:
    """
    Lớp phân loại Speech/Unvoiced/Voiced (SUV) sử dụng STE, ZCR và ST với ground truth
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
        
    def set_thresholds(self, ste_threshold: float, zcr_threshold: float, st_threshold: float):
        """
        Đặt ngưỡng từ bên ngoài (sau khi tối ưu với ground truth)
        
        Args:
            ste_threshold: Ngưỡng STE để phân biệt speech/silence
            zcr_threshold: Ngưỡng ZCR để phân biệt voiced/unvoiced
            st_threshold: Ngưỡng ST để phân biệt voiced/unvoiced
        """
        self.ste_thresholds['speech_silence'] = ste_threshold
        self.zcr_thresholds['voiced_unvoiced'] = zcr_threshold
        self.st_thresholds['voiced_unvoiced'] = st_threshold
        self.trained = True
        
    def classify(self, wav_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Phân loại SUV cho một file audio sử dụng STE + ZCR + ST (SUVDA)
        
        Args:
            wav_path (str): Đường dẫn file audio
            
        Returns:
            Tuple: (audio, ste, zcr, st, predictions)
        """
        if not self.trained:
            # Không cần huấn luyện, chỉ cần set ngưỡng từ bên ngoài
            pass
        
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
        
        # Lấy 3 ngưỡng chính
        energy_threshold = self.ste_thresholds.get('speech_silence', 0)
        zcr_threshold = self.zcr_thresholds.get('voiced_unvoiced', 0.5)  
        st_threshold = self.st_thresholds.get('voiced_unvoiced', 0.7)
        
        for i in range(min_length):
            # Logic theo bài báo SUVDA:
            # - Silence: STE < T_STE 
            # - Voiced: STE >= T_STE AND ST > T_ST AND ZCR < T_ZCR
            # - Unvoiced: STE >= T_STE AND (ST < T_ST OR ZCR > T_ZCR)
            
            if ste[i] < energy_threshold:
                # Energy thấp → SILENCE
                predictions[i] = 0
            else:
                # Energy cao → SPEECH
                # Phân biệt voiced vs unvoiced bằng ST và ZCR
                if st[i] > st_threshold and zcr[i] < zcr_threshold:
                    # ST cao (tần số thấp) + ZCR thấp → VOICED
                    predictions[i] = 1
                else:
                    # ST thấp hoặc ZCR cao → UNVOICED
                    predictions[i] = 2
        
        return audio, ste, zcr, st, predictions
        """
        Huấn luyện classifier với ground truth từ file lab
        
        Args:
            training_files: List of (wav_path, lab_path) tuples
            
        Returns:
            Dict: Thống kê training và ngưỡng tìm được
        """
        print("=== TRAINING SUV CLASSIFIER VỚI GROUND TRUTH ===")
        
        # Thu thập features và labels từ tất cả file
        all_ste_values = []
        all_zcr_values = []
        all_st_values = []
        all_labels = []
        
        print(f"Thu thập features và labels từ {len(training_files)} file...")
        
        for wav_path, lab_path in training_files:
            print(f"Xử lý file: {wav_path}")
            
            # Load audio và ground truth
            audio, _ = self.analyzer.load_audio(wav_path)
            segments = self.analyzer.load_labels(lab_path)
            
            # Tính features
            ste = self.analyzer.compute_ste(audio)
            zcr = self.analyzer.compute_zcr(audio)
            st = self.analyzer.compute_spectrum_tilt(audio)
            
            # Tạo frame labels từ segments
            frame_labels = self.analyzer.get_frame_labels(segments, len(audio))
            
            # Đảm bảo chiều dài khớp nhau
            min_length = min(len(ste), len(zcr), len(st), len(frame_labels))
            ste = ste[:min_length]
            zcr = zcr[:min_length]
            st = st[:min_length]
            frame_labels = frame_labels[:min_length]
            
            # Thu thập features và labels
            all_ste_values.extend(ste)
            all_zcr_values.extend(zcr)
            all_st_values.extend(st)
            all_labels.extend(frame_labels)
        
        print(f"Tổng cộng thu thập: {len(all_labels)} frames với ground truth")
        
        # Chuyển sang numpy arrays
        ste_array = np.array(all_ste_values)
        zcr_array = np.array(all_zcr_values)
        st_array = np.array(all_st_values)
        labels_array = np.array(all_labels)
        
        # Tính ngưỡng dựa trên ground truth
        training_stats = self._compute_supervised_thresholds(
            ste_array, zcr_array, st_array, labels_array
        )
        
        self.trained = True
        
        print(f"\\n=== NGƯỠNG TỪ GROUND TRUTH ===")
        print(f"Energy Threshold (STE): {self.ste_thresholds['speech_silence']:.6f}")
        print(f"ZCR Threshold: {self.zcr_thresholds['voiced_unvoiced']:.6f}")
        print(f"Spectrum Tilt Threshold: {self.st_thresholds['voiced_unvoiced']:.6f}")
        
        return {
            'training_stats': training_stats,
            'ste_thresholds': self.ste_thresholds,
            'zcr_thresholds': self.zcr_thresholds,
            'st_thresholds': self.st_thresholds
        }
    
    def _compute_supervised_thresholds(self, ste_values: np.ndarray, zcr_values: np.ndarray, 
                                     st_values: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Tính ngưỡng dựa trên ground truth labels
        Args:
            ste_values: STE features
            zcr_values: ZCR features  
            st_values: ST features
            labels: Ground truth labels (0=silence, 1=voiced, 2=unvoiced)
            
        Returns:
            Dict: Training statistics
        """
        print("\\n=== SUPERVISED THRESHOLD COMPUTATION ===")
        
        # Tách features theo class
        silence_mask = (labels == 0)
        voiced_mask = (labels == 1)
        unvoiced_mask = (labels == 2)
        speech_mask = (labels == 1) | (labels == 2)  # voiced + unvoiced
        
        print(f"Class distribution:")
        print(f"  Silence: {np.sum(silence_mask)} frames ({np.sum(silence_mask)/len(labels)*100:.1f}%)")
        print(f"  Voiced: {np.sum(voiced_mask)} frames ({np.sum(voiced_mask)/len(labels)*100:.1f}%)")
        print(f"  Unvoiced: {np.sum(unvoiced_mask)} frames ({np.sum(unvoiced_mask)/len(labels)*100:.1f}%)")
        
        # 1. STE Threshold: Tách silence vs speech
        silence_ste = ste_values[silence_mask]
        speech_ste = ste_values[speech_mask]
        
        if len(silence_ste) > 0 and len(speech_ste) > 0:
            # Optimal threshold: điểm giữa 95th percentile silence và 5th percentile speech
            silence_95th = np.percentile(silence_ste, 95)
            speech_5th = np.percentile(speech_ste, 5)
            ste_threshold = (silence_95th + speech_5th) / 2
        else:
            ste_threshold = np.percentile(ste_values, 25)
        
        self.ste_thresholds['speech_silence'] = ste_threshold
        
        # 2. ZCR Threshold: Tách voiced vs unvoiced (chỉ trong speech)
        voiced_zcr = zcr_values[voiced_mask]
        unvoiced_zcr = zcr_values[unvoiced_mask]
        
        if len(voiced_zcr) > 0 and len(unvoiced_zcr) > 0:
            # Optimal threshold: điểm giữa 95th percentile voiced và 5th percentile unvoiced
            voiced_95th = np.percentile(voiced_zcr, 95)
            unvoiced_5th = np.percentile(unvoiced_zcr, 5)
            zcr_threshold = (voiced_95th + unvoiced_5th) / 2
        else:
            zcr_threshold = np.median(zcr_values)
        
        self.zcr_thresholds['voiced_unvoiced'] = zcr_threshold
        
        # 3. ST Threshold: Tách voiced vs unvoiced (spectrum tilt)
        voiced_st = st_values[voiced_mask]
        unvoiced_st = st_values[unvoiced_mask]
        
        if len(voiced_st) > 0 and len(unvoiced_st) > 0:
            # Voiced có ST cao hơn (năng lượng ở tần số thấp), unvoiced có ST thấp hơn
            voiced_5th = np.percentile(voiced_st, 5)  # 5th percentile của voiced
            unvoiced_95th = np.percentile(unvoiced_st, 95)  # 95th percentile của unvoiced
            st_threshold = (voiced_5th + unvoiced_95th) / 2
        else:
            st_threshold = 0.7  # Default từ bài báo
        
        # Clamp ST threshold vào khoảng hợp lý
        st_threshold = max(0.3, min(0.9, st_threshold))
        self.st_thresholds['voiced_unvoiced'] = st_threshold
        
        print(f"\\nComputed thresholds:")
        print(f"  STE (Speech/Silence): {ste_threshold:.6f}")
        print(f"  ZCR (Voiced/Unvoiced): {zcr_threshold:.6f}")
        print(f"  ST (Voiced/Unvoiced): {st_threshold:.6f}")
        
        # Thống kê chi tiết
        training_stats = {
            'class_distribution': {
                'silence': int(np.sum(silence_mask)),
                'voiced': int(np.sum(voiced_mask)),
                'unvoiced': int(np.sum(unvoiced_mask))
            },
            'feature_statistics': {
                'ste': {
                    'silence': {'mean': np.mean(silence_ste) if len(silence_ste) > 0 else 0,
                               'std': np.std(silence_ste) if len(silence_ste) > 0 else 0},
                    'speech': {'mean': np.mean(speech_ste) if len(speech_ste) > 0 else 0,
                              'std': np.std(speech_ste) if len(speech_ste) > 0 else 0}
                },
                'zcr': {
                    'voiced': {'mean': np.mean(voiced_zcr) if len(voiced_zcr) > 0 else 0,
                              'std': np.std(voiced_zcr) if len(voiced_zcr) > 0 else 0},
                    'unvoiced': {'mean': np.mean(unvoiced_zcr) if len(unvoiced_zcr) > 0 else 0,
                                'std': np.std(unvoiced_zcr) if len(unvoiced_zcr) > 0 else 0}
                },
                'st': {
                    'voiced': {'mean': np.mean(voiced_st) if len(voiced_st) > 0 else 0,
                              'std': np.std(voiced_st) if len(voiced_st) > 0 else 0},
                    'unvoiced': {'mean': np.mean(unvoiced_st) if len(unvoiced_st) > 0 else 0,
                                'std': np.std(unvoiced_st) if len(unvoiced_st) > 0 else 0}
                }
            }
        }
        
        return training_stats
    
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
        
        # Bước 1: Load audio từ file WAV
        audio, _ = self.analyzer.load_audio(wav_path)
        
        # Bước 2: Tính các đặc trưng STE, ZCR và ST
        ste = self.analyzer.compute_ste(audio)      # Short-time Energy
        zcr = self.analyzer.compute_zcr(audio)      # Zero Crossing Rate  
        st = self.analyzer.compute_spectrum_tilt(audio)  # Spectrum Tilt
        
        # Bước 3: Đảm bảo chiều dài khớp nhau giữa các đặc trưng
        min_length = min(len(ste), len(zcr), len(st))
        ste = ste[:min_length]
        zcr = zcr[:min_length]
        st = st[:min_length]
        
        # Bước 4: Khởi tạo mảng predictions (0: silence, 1: voiced, 2: unvoiced)
        predictions = np.zeros(min_length, dtype=int)
        
        # Bước 5: Lấy 3 ngưỡng chính từ thuật toán SUVDA
        energy_threshold = self.ste_thresholds.get('speech_silence', 0)    # Ngưỡng STE
        zcr_threshold = self.zcr_thresholds.get('voiced_unvoiced', 0.5)    # Ngưỡng ZCR
        st_threshold = self.st_thresholds.get('voiced_unvoiced', 0.7)      # Ngưỡng ST
        
        # Bước 6: Áp dụng logic phân loại SUVDA cho từng frame
        for i in range(min_length):
            # Logic theo bài báo SUVDA:
            # - Silence: STE < T_STE (năng lượng thấp)
            # - Voiced: STE >= T_STE AND ST > T_ST AND ZCR < T_ZCR 
            #   (năng lượng cao + nhiều low freq + ít zero crossing)
            # - Unvoiced: STE >= T_STE AND (ST < T_ST OR ZCR > T_ZCR)
            #   (năng lượng cao + nhiều high freq hoặc nhiều zero crossing)
            
            if ste[i] < energy_threshold:
                # Điều kiện 1: Energy thấp → SILENCE
                predictions[i] = 0
            else:
                # Điều kiện 2: Energy cao → SPEECH
                # Phân biệt voiced vs unvoiced bằng ST và ZCR
                if st[i] > st_threshold and zcr[i] < zcr_threshold:
                    # ST cao (tần số thấp dominant) + ZCR thấp (ít thay đổi dấu) → VOICED
                    predictions[i] = 1
                else:
                    # ST thấp (tần số cao) hoặc ZCR cao (nhiều thay đổi dấu) → UNVOICED
                    predictions[i] = 2
        
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
        
        # Bước 1: Median filter tự viết để loại bỏ noise
        if len(smoothed) > 5:
            smoothed = self._median_filter(smoothed, kernel_size=5)
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
    
    def _median_filter(self, data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Args:
            data: Dữ liệu đầu vào
            kernel_size: Kích thước kernel (phải là số lẻ)
            
        Returns:
            np.ndarray: Dữ liệu đã được lọc
        """
        # Bước 1: Đảm bảo kernel_size là số lẻ
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Bước 2: Tính half_size để xác định vùng kernel
        half_size = kernel_size // 2
        filtered_data = np.zeros_like(data)
        
        # Bước 3: Áp dụng median filter cho từng điểm
        for i in range(len(data)):
            # Xác định vùng kernel xung quanh điểm i
            start = max(0, i - half_size)
            end = min(len(data), i + half_size + 1)
            
            # Lấy các giá trị trong kernel
            kernel_values = data[start:end]
            
            # Tính median bằng built-in function của numpy
            # Median: giá trị ở giữa khi sắp xếp tăng dần
            filtered_data[i] = np.median(kernel_values)
        
        return filtered_data
