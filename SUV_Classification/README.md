# SUV Classification System

Hệ thống phân loại tín hiệu âm thanh thành Speech/Unvoiced/Voiced (SUV) sử dụng Short-time Energy (STE) và Zero Crossing Rate (ZCR).

## Tính năng chính

- **Phân đoạn tự động**: Phân loại tín hiệu thành 3 loại:
  - `sil` (Silence): Khoảng lặng
  - `v` (Voiced): Tiếng nói hữu thanh
  - `uv` (Unvoiced): Tiếng nói vô thanh

- **🎯 Tối ưu ngưỡng tự động**: Grid search + Cross-validation
- **📈 Adaptive thresholding**: Ngưỡng thích ứng cho từng file
- **🔬 Thuật toán cải tiến**: STE + ZCR với nhiều enhancement
- **📊 Đánh giá toàn diện**: MAE, RMSE, accuracy, F1-score
- **📈 Trực quan hóa**: Biểu đồ chi tiết + so sánh với ground truth

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
SUV_Classification/
├── audio_analyzer.py        # Module phân tích âm thanh (STE/ZCR cải tiến)
├── suv_classifier.py        # Classifier chính với scoring system
├── evaluator.py            # Module đánh giá hiệu suất
├── threshold_optimizer.py  # 🎯 Tối ưu ngưỡng tự động
├── auto_optimize.py        # 🚀 Script tối ưu ngưỡng
├── main_optimized.py       # 📊 Script chính với ngưỡng tối ưu
├── requirements.txt       # Thư viện cần thiết
└── README.md             # Hướng dẫn này
```

## Sử dụng

### 🚀 Chạy với ngưỡng tự động tối ưu (KHUYẾN NGHỊ):

```bash
# Bước 1: Tối ưu ngưỡng tự động
python auto_optimize.py

# Bước 2: Chạy với ngưỡng đã tối ưu  
python main_optimized.py
```

## Quy trình hoạt động:

1. **auto_optimize.py**: Tìm ngưỡng tối ưu qua grid search + cross-validation
2. **main_optimized.py**: Sử dụng ngưỡng đã tối ưu để phân loại
3. Tạo báo cáo và biểu đồ trong thư mục `../results/`

## Thuật toán

### 🔬 STE (Short-time Energy) Cải tiến:
- **Pre-emphasis filter** (0.97) tăng cường tần số cao
- **Hamming window** giảm spectral leakage  
- **Log energy** tăng độ phân biệt
- **Z-score normalization** robust với outliers
- **Moving average** làm mịn tín hiệu

### 🔬 ZCR (Zero Crossing Rate) Cải tiến:
- **Pre-emphasis filter** (0.95) cho ZCR
- **Adaptive threshold** (2% max amplitude) tránh noise
- Chỉ tính ZCR trên **signal thực sự**, bỏ qua noise
- **Normalization + Moving average** với window lớn

### 🎯 Tìm ngưỡng tối ưu:
- **Grid search** trên multiple parameters
- **Cross-validation** để tránh overfitting  
- **Statistical separation** với 2-sigma rule
- **Adaptive thresholding** cho từng file

### 🧠 Thuật toán phân loại thông minh:
- **Scoring system** thay vì hard threshold
- Kết hợp thông tin từ **cả STE và ZCR**
- **Multi-step post-processing**: Median filter + Segment cleaning + Majority voting

## Tham số

- `frame_length`: 25ms (độ dài khung)
- `frame_shift`: 10ms (bước nhảy khung)
- `sr`: 16kHz (tần số lấy mẫu)
- `min_segment_length`: 30 frames (~300ms cho silence tối thiểu)

## Kết quả

Thư mục `results/` chứa:
- **🎯 Tối ưu ngưỡng:**
  - `best_thresholds.json`: Ngưỡng tối ưu tìm được
  - `threshold_optimization_report.txt`: Báo cáo chi tiết quá trình tối ưu
  - `detailed_optimization_results.json`: Kết quả chi tiết JSON

- **📊 Đánh giá hiệu suất:**
  - `optimized_evaluation_report.txt`: Báo cáo với ngưỡng tối ưu
  - `evaluation_report.txt`: Báo cáo với ngưỡng cơ bản
  - `training_statistics.txt`: Thống kê dữ liệu training

- **📈 Trực quan:**
  - `*_optimized_result.png`: Biểu đồ với ngưỡng tối ưu
  - `*_result.png`: Biểu đồ với ngưỡng cơ bản

## Định dạng dữ liệu

### File .wav
Tín hiệu âm thanh cần phân loại

### File .lab
Định dạng:
```
start_time<TAB>end_time<TAB>label
0.00    0.46    sil
0.46    1.39    v
1.39    1.50    uv
...
F0mean  122
F0std   18
```

## Đánh giá

- **Boundary Error**: MAE và RMSE giữa boundaries dự đoán và thực tế
- **Frame Accuracy**: Độ chính xác phân loại theo frame
- **Class Accuracy**: Độ chính xác cho từng class riêng biệt
