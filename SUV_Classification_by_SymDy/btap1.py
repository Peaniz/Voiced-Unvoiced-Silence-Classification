import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

class SymbolicDynamicsAnalyzer:
    def __init__(self, window_size=5, frame_size=80):
        """
        Symbolic Dynamics Analyzer theo phương pháp trong paper
        """
        self.window_size = window_size
        self.frame_size = frame_size
        
        # Các ngưỡng mặc định. Những ngưỡng này sẽ được cập nhật động
        # dựa trên dữ liệu đầu vào.
        self.thresholds = {
            'silence_energy_threshold': None,    
            'unvoiced_zcr_threshold': None,      
            'silence_symbol0_threshold': 0.85,   
            'voiced_entropy_max': 0.73,         
            'forbidden_words_voiced_max': 8,    
        }
        
        self.labels_map = {'sil': 'Silence', 'v': 'Voiced', 'uv': 'Unvoiced'}

        self.audio_data = None
        self.sample_rate = None
        self.analysis_results = None
    
    def convert_numpy_types(self, obj):
        """Đệ quy chuyển đổi các kiểu dữ liệu NumPy thành kiểu Python gốc."""
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_numpy_types(i) for i in obj]
        return obj
    
    def load_audio(self, wav_file_path):
        """Load audio file"""
        try:
            sample_rate, data = wavfile.read(wav_file_path)
            
            if data.ndim > 1:
                data = np.mean(data, axis=1)
                
            data = data.astype(np.float32)
            if np.max(np.abs(data)) > 0:
                data = data / np.max(np.abs(data))
                
            self.audio_data = data
            self.sample_rate = sample_rate
            
            print(f"✓ Loaded: {os.path.basename(wav_file_path)}")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Duration: {len(data)/sample_rate:.2f} seconds")
            print(f"  Total frames: {len(data)//self.frame_size}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading audio: {e}")
            return False
    
    def calculate_rms_energy(self, data):
        """Tính RMS energy cho silence detection"""
        return np.sqrt(np.mean(data ** 2))
    
    def symbolic_dynamics_transform(self, data):
        """Core symbolic dynamics transformation theo paper"""
        if np.std(data) > 0:
            data = (data - np.mean(data)) / np.std(data)
        
        symbol_string = []
        
        for i in range(len(data) - self.window_size + 1):
            window = data[i:i + self.window_size]
            sd = np.std(window)
            if sd == 0:
                sd = 1e-10
            
            threshold = 1.0 * sd
            symbol_value = 0
            
            for j in range(self.window_size - 1):
                diff = abs(window[j] - window[j+1])
                if diff <= threshold:
                    symbol_value += 1
            
            symbol_string.append(symbol_value)
            
        return symbol_string
    
    def get_symbol_histogram(self, symbol_string, normalize=True):
        """Tính histogram của symbols"""
        hist = {i: 0 for i in range(5)}
        for symbol in symbol_string:
            if 0 <= symbol <= 4:
                hist[symbol] += 1
        
        if normalize and len(symbol_string) > 0:
            total = len(symbol_string)
            hist = {k: v/total for k, v in hist.items()}
            
        return hist
    
    def get_word_histogram(self, symbol_string, word_length=2):
        """Tính word histogram length-2"""
        if len(symbol_string) < word_length:
            return {}
            
        words = []
        for i in range(len(symbol_string) - word_length + 1):
            word = tuple(symbol_string[i:i + word_length])
            words.append(word)
        
        word_hist = {}
        for word in words:
            base10_val = word[0] * 5 + word[1]
            word_hist[base10_val] = word_hist.get(base10_val, 0) + 1
            
        total = len(words)
        if total > 0:
            word_hist = {k: v/total for k, v in word_hist.items()}
            
        return word_hist
    
    def calculate_shannon_entropy(self, symbol_string):
        """Tính Shannon entropy"""
        hist = self.get_symbol_histogram(symbol_string, normalize=True)
        entropy = 0
        for prob in hist.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy
    
    def count_forbidden_words(self, symbol_string):
        """Đếm forbidden words"""
        word_hist = self.get_word_histogram(symbol_string, word_length=2)
        
        all_possible = set(range(25))
        observed = set(word_hist.keys())
        forbidden_count = len(all_possible - observed)
        
        for word_val, prob in word_hist.items():
            if prob <= 0.0015:
                forbidden_count += 1
        
        return forbidden_count
    
    def classify_segment(self, frame_features):
        """Phân loại segment sử dụng các ngưỡng đã được tính toán"""
        
        rms_energy = frame_features['rms_energy']
        zcr = frame_features['zcr']
        entropy = frame_features['entropy']
        forbidden_words = frame_features['forbidden_words']
        
        # 1. Phân loại Silence dựa trên Energy và Symbol0
        if rms_energy < self.thresholds['silence_energy_threshold']:
            return "sil", 0.95
        
        # 2. Phân loại Unvoiced dựa trên ZCR và Entropy
        if zcr > self.thresholds['unvoiced_zcr_threshold'] and entropy > 0.8:
            return "uv", 0.9
        
        # 3. Mặc định là Voiced
        return "v", 0.85
    
    def calculate_energy(self, frames):
        """Tính toán năng lượng của từng khung"""
        return np.sum(frames ** 2, axis=1)

    def calculate_zcr(self, frames):
        """Tính toán Zero-Crossing Rate của từng khung"""
        zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / (2 * self.frame_size)
        return zcr

    def auto_set_all_thresholds(self, all_features):
        """Tự động xác định tất cả các ngưỡng cần thiết từ dữ liệu thực tế"""
        energy = np.array([f['rms_energy'] for f in all_features])
        zcr = np.array([f['zcr'] for f in all_features])
        
        # 1. Ngưỡng năng lượng cho Silence
        sorted_energy = np.sort(energy)
        num_frames_for_baseline = int(0.1 * len(sorted_energy))
        if num_frames_for_baseline == 0: num_frames_for_baseline = 1
        baseline_energy = np.mean(sorted_energy[:num_frames_for_baseline])
        self.thresholds['silence_energy_threshold'] = baseline_energy * 3.0
        
        # 2. Ngưỡng ZCR cho Unvoiced
        # Loại bỏ các khung có năng lượng rất thấp (silence) để tính ZCR
        non_silent_zcr = zcr[energy > self.thresholds['silence_energy_threshold']]
        if len(non_silent_zcr) > 0:
            median_zcr = np.median(non_silent_zcr)
            self.thresholds['unvoiced_zcr_threshold'] = median_zcr * 1.5
        else:
            self.thresholds['unvoiced_zcr_threshold'] = 0.2 # Fallback
        
        print(f"✓ Ngưỡng năng lượng silence tự động: {self.thresholds['silence_energy_threshold']:.6f}")
        print(f"✓ Ngưỡng ZCR unvoiced tự động: {self.thresholds['unvoiced_zcr_threshold']:.4f}")

    def analyze_audio(self):
        """Phân tích toàn bộ audio"""
        if self.audio_data is None:
            print("No audio data loaded!")
            return
            
        num_frames = len(self.audio_data) // self.frame_size
        print(f"Analyzing {num_frames} frames...")

        frames = np.array([
            self.audio_data[i * self.frame_size:(i + 1) * self.frame_size] 
            for i in range(num_frames)
        ])
        
        # Bước 1: Trích xuất tất cả các đặc trưng
        all_features = []
        for i in range(num_frames):
            frame_data = frames[i]
            symbol_string = self.symbolic_dynamics_transform(frame_data)
            
            features = {
                'frame_idx': i,
                'start_time': i * self.frame_size / self.sample_rate,
                'end_time': (i + 1) * self.frame_size / self.sample_rate,
                'rms_energy': self.calculate_rms_energy(frame_data),
                'zcr': self.calculate_zcr(frame_data.reshape(1, -1))[0],
                'symbol_string': symbol_string,
                'entropy': self.calculate_shannon_entropy(symbol_string),
                'forbidden_words': self.count_forbidden_words(symbol_string),
            }
            all_features.append(features)
        
        # Bước 2: Dùng các đặc trưng đã trích xuất để tính toán ngưỡng tự động
        self.auto_set_all_thresholds(all_features)
        
        # Bước 3: Dùng các ngưỡng tự động để phân loại
        for features in all_features:
            classification, confidence = self.classify_segment(features)
            features['classification'] = classification
            features['confidence'] = confidence

        self.analysis_results = all_features

        classifications = [r['classification'] for r in all_features]
        unique, counts = np.unique(classifications, return_counts=True)
        
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total frames: {num_frames}")
        print(f"Duration: {len(self.audio_data)/self.sample_rate:.2f} seconds")
        
        print("\nClassification Distribution:")
        for cls, count in zip(unique, counts):
            percentage = count / num_frames * 100
            cls_name = self.labels_map.get(cls, cls)
            print(f"  {cls_name}: {count} frames ({percentage:.1f}%)")
        
        return all_features
    
    def visualize_main_analysis(self, save_folder=None):
        if self.analysis_results is None:
            print("No analysis results available!")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 1, figure=fig, hspace=0.4)
        
        # 1. Biểu đồ tổng quát âm thanh
        ax1 = fig.add_subplot(gs[0])
        time_axis = np.arange(len(self.audio_data)) / self.sample_rate
        ax1.plot(time_axis, self.audio_data, color='blue', linewidth=0.8, alpha=0.7)
        ax1.set_title('A. Original Audio Signal', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1.1, 1.1)
        
        # 2. Biểu đồ symbolic dynamics transformation (cải tiến)
        ax2 = fig.add_subplot(gs[1])
        colors_map = {'sil': 'lightblue', 'v': 'red', 'uv': 'orange'}
        
        # Tô màu biểu đồ dựa trên phân loại
        for result in self.analysis_results:
            symbols = result['symbol_string']
            start_time = result['start_time']
            end_time = result['end_time']
            classification = result['classification']
            
            if symbols:
                symbol_times = np.linspace(start_time, end_time, len(symbols))
                color = colors_map.get(classification, 'gray')
                ax2.step(symbol_times, symbols, where='post', color=color, linewidth=1.5)
                
        full_symbol_seq = []
        full_time_seq = []
        for result in self.analysis_results:
            symbols = result['symbol_string']
            start_time = result['start_time']
            end_time = result['end_time']
            if symbols:
                full_symbol_seq.extend(symbols)
                full_time_seq.extend(np.linspace(start_time, end_time, len(symbols)))
        
        ax2.scatter(full_time_seq[::10], full_symbol_seq[::10], color='black', s=10, alpha=0.6)
        
        ax2.set_title('B. Symbolic Dynamics Transformation (Color-Coded by Classification)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Symbol Value (0-4)')
        ax2.set_ylim(-0.5, 4.5)
        ax2.set_yticks([0, 1, 2, 3, 4])
        ax2.grid(True, alpha=0.3)
        
        ax2.text(0.02, 0.98, 'Symbols: 0=Low variability, 4=High variability', 
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        legend_elements = [patches.Patch(color=color, label=self.labels_map[cls]) 
                           for cls, color in colors_map.items()]
        ax2.legend(handles=legend_elements, loc='upper right')

        # 3. Kết quả phân loại cuối cùng
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(time_axis, self.audio_data, color='lightgray', linewidth=0.5, alpha=0.6, label='Original signal')
        
        # Color-code segments
        colors = {'sil': 'lightblue', 'v': 'red', 'uv': 'orange'}
        
        # Track segments for legend
        legend_added = set()
        
        for result in self.analysis_results:
            start_time = result['start_time']
            end_time = result['end_time']
            classification = result['classification']
            confidence = result['confidence']
            
            # Transparency based on confidence
            alpha = 0.4 + 0.4 * confidence
            color = colors.get(classification, 'gray')
            
            # Add to legend only once per class
            label = self.labels_map[classification] if classification not in legend_added else None
            if label:
                legend_added.add(classification)
            
            ax3.axvspan(start_time, end_time, alpha=alpha, color=color, label=label)
        
        ax3.set_title('C. Final Classification Results (Silence/Voiced/Unvoiced)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Amplitude')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        ax3.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        
        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            plt.savefig(os.path.join(save_folder, 'main_analysis.png'), 
                        dpi=300, bbox_inches='tight')
            print(f"✓ Main analysis saved to {save_folder}")
        
        plt.show()
    
    def show_detailed_analysis(self, save_folder=None):
        """Hiển thị các biểu đồ chi tiết trong cửa sổ riêng"""
        if self.analysis_results is None:
            print("No analysis results available!")
            return
        
        # Window 1: Feature distributions
        fig1 = plt.figure(figsize=(15, 10))
        gs1 = GridSpec(2, 2, figure=fig1, hspace=0.4, wspace=0.3)
        
        # Extract data for plotting
        classifications = [r['classification'] for r in self.analysis_results]
        entropies = [r['entropy'] for r in self.analysis_results]
        forbidden_words = [r['forbidden_words'] for r in self.analysis_results]
        energies = [r['rms_energy'] for r in self.analysis_results]
        
        colors_map = {'sil': 'lightblue', 'v': 'red', 'uv': 'orange'}
        colors = [colors_map[c] for c in classifications]
        
        # Entropy distribution
        ax1 = fig1.add_subplot(gs1[0, 0])
        ax1.scatter(range(len(entropies)), entropies, c=colors, alpha=0.6, s=20)
        ax1.set_title('Shannon Entropy Distribution')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Entropy')
        ax1.grid(True, alpha=0.3)
        
        # Forbidden words distribution
        ax2 = fig1.add_subplot(gs1[0, 1])
        ax2.scatter(range(len(forbidden_words)), forbidden_words, c=colors, alpha=0.6, s=20)
        ax2.set_title('Forbidden Words Distribution')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Forbidden Words Count')
        ax2.grid(True, alpha=0.3)
        
        # RMS Energy distribution
        ax3 = fig1.add_subplot(gs1[1, 0])
        ax3.scatter(range(len(energies)), energies, c=colors, alpha=0.6, s=20)
        ax3.set_title('RMS Energy Distribution')
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('RMS Energy')
        ax3.grid(True, alpha=0.3)
        
        # Feature space (2D)
        ax4 = fig1.add_subplot(gs1[1, 1])
        ax4.scatter(entropies, forbidden_words, c=colors, alpha=0.6, s=30)
        ax4.set_xlabel('Shannon Entropy')
        ax4.set_ylabel('Forbidden Words')
        ax4.set_title('Feature Space Analysis')
        ax4.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [patches.Patch(color=color, label=self.labels_map[cls]) 
                           for cls, color in colors_map.items()]
        ax4.legend(handles=legend_elements)
        
        plt.tight_layout()
        if save_folder:
            plt.savefig(os.path.join(save_folder, 'detailed_analysis.png'), 
                        dpi=300, bbox_inches='tight')
        plt.show()
        
        # Window 2: Symbol histograms comparison
        fig2 = plt.figure(figsize=(15, 8))
        gs2 = GridSpec(2, 3, figure=fig2, hspace=0.4, wspace=0.3)
        
        # Aggregate symbol histograms by class
        class_symbols = {'sil': [], 'v': [], 'uv': []}
        for result in self.analysis_results:
            cls = result['classification']
            symbols = result['symbol_string']
            class_symbols[cls].extend(symbols)
        
        # Plot histograms for each class
        class_titles = {'sil': 'Silence', 'v': 'Voiced', 'uv': 'Unvoiced'}
        
        for i, (cls, symbols) in enumerate(class_symbols.items()):
            if symbols:  # Only plot if we have data
                ax = fig2.add_subplot(gs2[0, i])
                
                # Calculate histogram
                hist_data = [0] * 5
                for s in symbols:
                    if 0 <= s <= 4:
                        hist_data[s] += 1
                
                # Normalize
                total = sum(hist_data)
                if total > 0:
                    hist_data = [h/total for h in hist_data]
                
                bars = ax.bar(range(5), hist_data, color=colors_map[cls], alpha=0.7)
                ax.set_title(f'{class_titles[cls]} - Symbol Distribution')
                ax.set_xlabel('Symbol')
                ax.set_ylabel('Relative Frequency')
                ax.set_xticks(range(5))
                ax.grid(True, alpha=0.3)
                
                # Add values on bars
                for j, (bar, freq) in enumerate(zip(bars, hist_data)):
                    if freq > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{freq:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Bottom row: Statistics summary
        ax_stats = fig2.add_subplot(gs2[1, :])
        ax_stats.axis('off')
        
        # Calculate statistics for each class
        stats_text = "CLASSIFICATION STATISTICS:\n\n"
        for cls in ['sil', 'v', 'uv']:
            cls_results = [r for r in self.analysis_results if r['classification'] == cls]
            if cls_results:
                avg_entropy = np.mean([r['entropy'] for r in cls_results])
                avg_forbidden = np.mean([r['forbidden_words'] for r in cls_results])
                avg_energy = np.mean([r['rms_energy'] for r in cls_results])
                count = len(cls_results)
                
                stats_text += f"{self.labels_map.get(cls, cls)} ({count} frames):\n"
                stats_text += f"  Avg Entropy: {avg_entropy:.3f}\n"
                stats_text += f"  Avg Forbidden Words: {avg_forbidden:.1f}\n"
                stats_text += f"  Avg RMS Energy: {avg_energy:.4f}\n\n"
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                      fontsize=12, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        if save_folder:
            plt.savefig(os.path.join(save_folder, 'symbol_histograms.png'), 
                        dpi=300, bbox_inches='tight')
        plt.show()

    def export_lab_file(self, save_folder, filename_prefix="output"):
        if self.analysis_results is None:
            print("No analysis results available!")
            return
        
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Merge consecutive frames with same classification
        segments = []
        current_class = None
        current_start = None
        
        for result in self.analysis_results:
            if current_class != result['classification']:
                # Save previous segment
                if current_class is not None:
                    segments.append((current_start, result['start_time'], current_class))
                
                # Start new segment
                current_class = result['classification']
                current_start = result['start_time']
        
        # Add final segment
        if current_class is not None:
            segments.append((current_start, self.analysis_results[-1]['end_time'], current_class))
        
        # Write .lab file
        lab_filename = os.path.join(save_folder, f"{filename_prefix}.lab")
        
        with open(lab_filename, 'w') as f:
            for start, end, cls in segments:
                f.write(f"{start:.2f}\t{end:.2f}\t{self.labels_map.get(cls, cls)}\n")
            
            voiced_segments = [s for s in self.analysis_results if s['classification'] == 'v']
            if voiced_segments:
                f0_mean = 150
                f0_std = 25
            else:
                f0_mean = 0
                f0_std = 0
            
            f.write(f"F0mean\t{f0_mean}\n")
            f.write(f"F0std\t{f0_std}\n")
        
        print(f"✓ Lab file exported: {lab_filename}")
        
        json_filename = os.path.join(save_folder, f"{filename_prefix}_detailed.json")
        with open(json_filename, 'w') as f:
            serializable_results = self.convert_numpy_types(self.analysis_results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"✓ Detailed results exported: {json_filename}")
        
        return lab_filename

def run_analysis(wav_file_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"analysis_output_{timestamp}"
    
    analyzer = SymbolicDynamicsAnalyzer()
    
    if analyzer.load_audio(wav_file_path):
        print("\n1. Running symbolic dynamics analysis...")
        analyzer.analyze_audio()
        
        print("\n2. Showing main visualization...")
        analyzer.visualize_main_analysis(save_folder)
        
        print("\n3. Showing detailed analysis...")
        analyzer.show_detailed_analysis(save_folder)
        
        print("\n4. Exporting results...")
        filename_prefix = os.path.splitext(os.path.basename(wav_file_path))[0]
        analyzer.export_lab_file(save_folder, filename_prefix)
        
        print(f"\nAnalysis complete! All results saved to: {save_folder}")
        return analyzer

# Example usage
if __name__ == "__main__":
    analyzer = run_analysis('file/studio_F1.wav')