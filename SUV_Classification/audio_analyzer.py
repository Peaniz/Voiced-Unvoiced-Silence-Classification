import numpy as np
import matplotlib.pyplot as plt
import os
import wave
from typing import List, Tuple, Dict

class AudioAnalyzer:
    """
    Lớp phân tích tín hiệu âm thanh để phân loại Speech/Unvoiced/Voiced (SUV)
    sử dụng Short-time Energy (STE) và Zero Crossing Rate (ZCR)
    """
    
    def __init__(self, frame_length=0.025, frame_shift=0.01, sr=16000):
        """
        Khởi tạo AudioAnalyzer
        Args:
            frame_length (float): Độ dài khung (giây), mặc định 25ms
            frame_shift (float): Bước nhảy khung (giây), mặc định 10ms  
            sr (int): Tần số lấy mẫu, mặc định 16kHz
        """
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.sr = sr
        self.frame_size = int(frame_length * sr)
        self.hop_size = int(frame_shift * sr)
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Đọc file audio WAV sử dụng built-in wave module
        
        Args:
            audio_path (str): Đường dẫn file audio
            
        Returns:
            Tuple[np.ndarray, int]: Tín hiệu và sample rate
        """
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                # Lấy thông tin file
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                channels = wav_file.getnchannels()
                
                # Đọc raw data
                raw_data = wav_file.readframes(frames)
                
                # Chuyển đổi bytes thành numpy array
                if sample_width == 1:
                    # 8-bit unsigned
                    audio = np.frombuffer(raw_data, dtype=np.uint8)
                    audio = (audio.astype(np.float32) - 128) / 128.0
                elif sample_width == 2:
                    # 16-bit signed
                    audio = np.frombuffer(raw_data, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32768.0
                elif sample_width == 4:
                    # 32-bit
                    audio = np.frombuffer(raw_data, dtype=np.int32)
                    audio = audio.astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                # Nếu stereo, chuyển thành mono bằng cách lấy trung bình
                if channels == 2:
                    audio = audio.reshape(-1, 2)
                    audio = np.mean(audio, axis=1)
                
                # Resample nếu cần (simple decimation/interpolation)
                if sample_rate != self.sr:
                    audio = self._simple_resample(audio, sample_rate, self.sr)
                
                return audio, self.sr
                
        except Exception as e:
            raise ValueError(f"Không thể đọc file audio {audio_path}: {e}")
    
    def _simple_resample(self, audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample đơn giản bằng decimation/interpolation
        
        Args:
            audio: Tín hiệu gốc
            original_sr: Sample rate gốc
            target_sr: Sample rate đích
            
        Returns:
            np.ndarray: Tín hiệu đã resample
        """
        if original_sr == target_sr:
            return audio
        
        # Tính tỷ lệ resample
        ratio = target_sr / original_sr
        
        if ratio == 1.0:
            return audio
        elif ratio < 1.0:
            # Downsampling - decimation
            step = int(1 / ratio)
            return audio[::step]
        else:
            # Upsampling - linear interpolation đơn giản
            factor = int(ratio)
            upsampled = np.zeros(len(audio) * factor)
            upsampled[::factor] = audio
            
            # Linear interpolation cho các sample trung gian
            for i in range(1, factor):
                upsampled[i::factor] = (audio * (factor - i) / factor + 
                                      np.roll(audio, -1) * i / factor)[:-1]
            
            return upsampled[:int(len(audio) * ratio)]
    
    def load_labels(self, label_path: str) -> List[Dict]:
        """
        Đọc file label .lab
        
        Args:
            label_path (str): Đường dẫn file label
            
        Returns:
            List[Dict]: Danh sách các đoạn với thời gian và nhãn
        """
        segments = []
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line and not line.startswith('F0'):
                parts = line.split('\t')
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    label = parts[2].strip()
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'label': label
                    })
        return segments
    
    def compute_ste(self, audio: np.ndarray) -> np.ndarray:
        """
        Tính Short-time Energy (STE) với các cải tiến - TỰ VIẾT
        
        Args:
            audio (np.ndarray): Tín hiệu âm thanh
            
        Returns:
            np.ndarray: Mảng STE của các khung
        """
        # Bước 1: Áp dụng pre-emphasis filter để tăng cường tần số cao
        # Công thức: y[n] = x[n] - α*x[n-1] với α = 0.97
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Bước 2: Chia tín hiệu thành các khung với frame_size và hop_size
        # Số khung = (tổng_số_mẫu - kích_thước_khung) / bước_nhảy + 1
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        frames = np.zeros((self.frame_size, num_frames))
        
        # Chia tín hiệu thành các khung chồng lấp
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            if end <= len(audio):
                frames[:, i] = audio[start:end]
        
        # Bước 3: Áp dụng Hamming window để giảm spectral leakage
        # w[n] = 0.54 - 0.46*cos(2πn/(N-1))
        window = np.hamming(self.frame_size)
        windowed_frames = frames * window[:, np.newaxis]
        
        # Bước 4: Tính energy cho mỗi khung
        # STE[i] = Σ(x²[n]) với n trong khung i
        ste = np.sum(windowed_frames**2, axis=0)
        
        # Bước 5: Tránh log(0) bằng cách thêm epsilon
        epsilon = 1e-12
        ste = ste + epsilon
        
        # Bước 6: Chuyển sang log energy để tăng độ phân biệt
        # Log energy: log10(STE[i])
        log_ste = np.log10(ste)
        
        # Bước 7: Normalize log STE về mean=0, std=1
        # Normalized = (x - mean) / std
        log_ste = (log_ste - np.mean(log_ste)) / (np.std(log_ste) + epsilon)
        
        # Bước 8: Áp dụng moving average để làm mịn
        # Moving average với window size = 3
        window_size = 3
        if len(log_ste) > window_size:
            smoothed_ste = np.convolve(log_ste, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed_ste = log_ste
            
        return smoothed_ste
    
    def compute_zcr(self, audio: np.ndarray) -> np.ndarray:
        """
        Tính Zero Crossing Rate (ZCR) với các cải tiến - TỰ VIẾT
        
        Args:
            audio (np.ndarray): Tín hiệu âm thanh
            
        Returns:
            np.ndarray: Mảng ZCR của các khung
        """
        # Bước 1: Áp dụng pre-emphasis filter nhẹ cho ZCR
        # Pre-emphasis nhẹ hơn so với STE
        pre_emphasis = 0.95
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Bước 2: Chia tín hiệu thành các khung
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        frames = np.zeros((self.frame_size, num_frames))
        
        # Tạo các khung chồng lấp
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            if end <= len(audio):
                frames[:, i] = audio[start:end]
        
        # Bước 3: Tính ZCR cho mỗi khung với cải tiến
        zcr = np.zeros(frames.shape[1])
        
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            # Bước 4: Áp dụng threshold để tránh nhiễu nhỏ ảnh hưởng
            # Threshold = 2% của max amplitude trong frame
            threshold = 0.02 * np.max(np.abs(frame))
            
            # Bước 5: Chỉ tính zero crossing khi tín hiệu vượt qua threshold
            # Tạo tín hiệu đã được threshold
            thresholded_frame = np.where(np.abs(frame) > threshold, frame, 0)
            
            # Bước 6: Tính zero crossing rate
            if np.any(thresholded_frame != 0):
                # Tìm các điểm khác 0
                non_zero_indices = np.where(thresholded_frame != 0)[0]
                if len(non_zero_indices) > 1:
                    # Chỉ tính ZCR trên các segment có tín hiệu
                    # ZCR = số lần đổi dấu / tổng số điểm
                    signs = np.sign(thresholded_frame[non_zero_indices])
                    sign_changes = np.sum(np.abs(np.diff(signs))) / 2
                    zcr[i] = sign_changes / len(non_zero_indices)
                else:
                    zcr[i] = 0
            else:
                zcr[i] = 0
        
        # Bước 7: Normalize ZCR về [0,1]
        max_zcr = np.max(zcr)
        if max_zcr > 0:
            zcr = zcr / max_zcr
        
        # Bước 8: Áp dụng moving average để làm mịn
        # Window size lớn hơn một chút cho ZCR
        window_size = 5
        if len(zcr) > window_size:
            smoothed_zcr = np.convolve(zcr, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed_zcr = zcr
            
        return smoothed_zcr
    
    def compute_spectrum_tilt(self, audio: np.ndarray) -> np.ndarray:
        """
        Tính Spectrum Tilt (ST)
        ST > 0.7: Voiced (năng lượng mạnh ở tần số thấp)
        ST < 0.7: Unvoiced (năng lượng nhiều ở tần số cao)
        
        Args:
            audio (np.ndarray): Tín hiệu âm thanh
            
        Returns:
            np.ndarray: Mảng ST của các khung
        """
        # Bước 1: Pre-emphasis filter nhẹ cho ST
        pre_emphasis = 0.95
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        # Bước 2: Chia tín hiệu thành các khung
        num_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        frames = np.zeros((self.frame_size, num_frames))
        
        # Tạo các khung chồng lấp
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            if end <= len(audio):
                frames[:, i] = audio[start:end]
        
        # Bước 3: Áp dụng Hamming window
        window = np.hamming(self.frame_size)
        windowed_frames = frames * window[:, np.newaxis]
        
        # Bước 4: Tính Spectrum Tilt cho mỗi khung
        st = np.zeros(frames.shape[1])
        
        for i in range(frames.shape[1]):
            frame = windowed_frames[:, i]
            
            # Bước 5: Tính FFT để có magnitude spectrum
            # FFT: X[k] = Σ x[n] * e^(-j*2π*k*n/N)
            fft_frame = np.fft.rfft(frame)
            magnitude_spectrum = np.abs(fft_frame)
            
            # Bước 6: Tránh log(0) bằng epsilon
            magnitude_spectrum = magnitude_spectrum + 1e-12
            log_spectrum = np.log10(magnitude_spectrum)
            
            # Bước 7: Tần số tương ứng với mỗi bin
            freqs = np.fft.rfftfreq(len(frame), 1/self.sr)
            
            # Bước 8: Chia spectrum thành 2 phần: low freq và high freq
            # Low freq: 0-1000Hz (voiced energy), High freq: 1000Hz-Nyquist (unvoiced energy)
            low_freq_cutoff = 1000  # Hz
            
            # Tìm index tương ứng với cutoff frequency
            cutoff_idx = int(low_freq_cutoff * len(freqs) / (self.sr/2))
            cutoff_idx = max(1, min(cutoff_idx, len(freqs)-1))
            
            # Bước 9: Tính năng lượng trung bình cho từng dải
            # Low frequency energy (0-1000Hz)
            low_freq_energy = np.mean(magnitude_spectrum[1:cutoff_idx]**2)
            # High frequency energy (1000Hz-Nyquist)
            high_freq_energy = np.mean(magnitude_spectrum[cutoff_idx:]**2)
            
            # Bước 10: Tính Spectrum Tilt
            # ST = Low_Energy / High_Energy
            if high_freq_energy > 0:
                st_raw = low_freq_energy / high_freq_energy
                # Normalize về [0,1] để dễ sử dụng ngưỡng 0.7
                # Sigmoid-like normalization: x/(x+1)
                st[i] = min(1.0, st_raw / (st_raw + 1.0))
            else:
                st[i] = 1.0  # Toàn bộ năng lượng ở low freq → Voiced
        
        # Bước 11: Làm mịn ST bằng moving average
        window_size = 3
        if len(st) > window_size:
            smoothed_st = np.convolve(st, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed_st = st
            
        return smoothed_st
    
    def get_frame_labels(self, segments: List[Dict], audio_length: int) -> np.ndarray:
        """
        Chuyển đổi segment labels thành frame labels
        
        Args:
            segments (List[Dict]): Danh sách segments
            audio_length (int): Độ dài tín hiệu
            
        Returns:
            np.ndarray: Nhãn cho mỗi khung (0: sil, 1: v, 2: uv)
        """
        num_frames = (audio_length - self.frame_size) // self.hop_size + 1
        frame_labels = np.zeros(num_frames, dtype=int)
        
        for segment in segments:
            start_frame = int(segment['start'] * self.sr // self.hop_size)
            end_frame = int(segment['end'] * self.sr // self.hop_size)
            
            # Đảm bảo không vượt quá số frame
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)
            
            if segment['label'] == 'sil':
                label_value = 0
            elif segment['label'] == 'v':
                label_value = 1
            elif segment['label'] == 'uv':
                label_value = 2
            else:
                label_value = 0  # Mặc định là silence
                
            frame_labels[start_frame:end_frame] = label_value
            
        return frame_labels
    
    def compute_statistics(self, feature_values: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Tính thống kê cho các đặc trưng theo nhãn
        
        Args:
            feature_values (np.ndarray): Giá trị đặc trưng
            labels (np.ndarray): Nhãn tương ứng
            
        Returns:
            Dict: Thống kê mean và std cho mỗi class
        """
        stats = {}
        
        for label_value, label_name in [(0, 'silence'), (1, 'voiced'), (2, 'unvoiced')]:
            mask = labels == label_value
            if np.sum(mask) > 0:
                values = feature_values[mask]
                stats[label_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
            else:
                stats[label_name] = {
                    'mean': 0,
                    'std': 0,  
                    'count': 0
                }
                
        return stats
