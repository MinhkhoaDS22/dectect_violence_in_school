# 🎓 Detect Violence in School

> **Đồ án tốt nghiệp** — Hệ thống phát hiện hành vi bạo lực trong trường học sử dụng Deep Learning.

Hệ thống sử dụng pipeline kết hợp **YOLO** (phát hiện & tracking người) + **CNN-BiLSTM-Attention** (phân loại hành vi bạo lực theo thời gian) để phân tích video giám sát và phát hiện các hành vi bạo lực một cách tự động.

---

## 📋 Mục lục

- [Tổng quan Pipeline](#-tổng-quan-pipeline)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Chi tiết Pipeline](#-chi-tiết-pipeline)
- [Kết quả](#-kết-quả)

---

## 🔄 Tổng quan Pipeline

```
Video giám sát
       │
       ▼
┌──────────────────────────┐
│  1. YOLO Detection       │  dectect_people.py
│     + ByteTrack Tracking │  → Phát hiện & theo dõi người
│     → Xuất XML (CVAT)    │  → Bounding box mỗi frame
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  2. Tiền xử lý           │  train_ai.py (phần 1)
│     • Chuẩn hóa FPS→30   │
│     • Chuẩn hóa Size→64  │
│     • Normalize [0,1]     │
│     • Sliding Window      │
│       30 frame, step 5    │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  3. Huấn luyện Model     │  train_ai.py (phần 2)
│     CNN → Conv1D →        │
│     BiLSTM → Attention →  │
│     Classifier            │
│     + MixUp, Augmentation │
└──────────┬───────────────┘
           │
           ▼
    ┌──────────────┐
    │  Violence?   │
    │  ✅ / ❌     │
    └──────────────┘
```

---

## 📁 Cấu trúc dự án

```
dectect_violence_in_school/
│
├── train_ai.py            # 🧠 Script chính: tiền xử lý + huấn luyện model
├── dectect_people.py      # 👤 YOLO detection + tracking → xuất XML annotation
├── check_labels.py        # 🔍 Kiểm tra & preview annotation đã sửa
├── rename.py              # 📝 Tiện ích đổi tên video theo format chuẩn
│
├── data/                  # 📂 Dataset video (không đưa lên GitHub)
│   ├── violence/          #    200 video bạo lực
│   └── non_violence/      #    200 video không bạo lực
│
├── fix_labels/            # 🏷️ Annotation XML đã chỉnh sửa (CVAT format)
│   ├── violence/
│   └── non_violence/
│
├── processed_tracks_v4/   # 💾 Cache dữ liệu đã tiền xử lý (tự sinh)
│   ├── manifest.json
│   ├── violence/*.npy
│   └── non_violence/*.npy
│
├── .gitignore
└── README.md
```

---

## 💻 Yêu cầu hệ thống

- **Python** 3.9+
- **RAM**: tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: khuyến nghị NVIDIA GPU với CUDA (chạy được trên CPU nhưng chậm)

### Thư viện Python

| Thư viện | Mục đích |
|----------|----------|
| `torch` | Deep learning framework |
| `torchvision` | Model pretrained (nếu dùng) |
| `opencv-python` | Xử lý video & ảnh |
| `ultralytics` | YOLOv8/v11 detection |
| `numpy` | Tính toán mảng |
| `matplotlib` | Vẽ biểu đồ training |
| `seaborn` | Confusion matrix |
| `scikit-learn` | Metrics đánh giá |
| `tqdm` | Progress bar |

---

## 🛠 Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/dectect_violence_in_school.git
cd dectect_violence_in_school

# 2. Cài đặt dependencies
pip install torch torchvision opencv-python ultralytics numpy matplotlib seaborn scikit-learn tqdm
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Chuẩn bị Dataset

Đặt video vào thư mục `data/`:
```
data/
├── violence/        # Video có bạo lực
│   ├── v_001.mp4
│   ├── v_002.mp4
│   └── ...
└── non_violence/    # Video bình thường
    ├── nv_001.mp4
    ├── nv_002.mp4
    └── ...
```

### Bước 2: Detection & Tracking với YOLO

```bash
python dectect_people.py
```

Script sẽ:
- Dùng **YOLOv11s** phát hiện người trong mỗi frame
- Dùng **ByteTrack** theo dõi (tracking) mỗi người qua các frame
- Xuất ra annotation XML theo chuẩn **CVAT** vào `labels/`
- Xuất video preview có vẽ bounding box vào `previews/`

### Bước 3: Kiểm tra & sửa Annotation (nếu cần)

```bash
# Sửa bounding box bằng CVAT hoặc thủ công
# Copy file XML đã sửa vào fix_labels/

# Kiểm tra kết quả:
python check_labels.py
```

### Bước 4: Huấn luyện Model

```bash
python train_ai.py
```

Script sẽ tự động chạy toàn bộ pipeline:
1. **Tiền xử lý** — Chuẩn hóa FPS, Size, tạo Sliding Window, lưu cache ra disk
2. **Huấn luyện** — CNN-BiLSTM-Attention với MixUp, augmentation, early stopping
3. **Đánh giá** — Xuất confusion matrix, classification report, biểu đồ training
4. **Lưu model** — `best_model_v2.pth`

---

## 🔬 Chi tiết Pipeline

### 1. YOLO Detection + Tracking (`dectect_people.py`)

| Thành phần | Chi tiết |
|------------|----------|
| Model | YOLOv11s |
| Tracker | ByteTrack |
| Confidence | 0.38 |
| IoU threshold | 0.55 |
| Output | XML annotation chuẩn CVAT |

### 2. Tiền xử lý dữ liệu (`train_ai.py` — phần 1)

Pipeline chuẩn hóa 4 bước:

| Bước | Mô tả | Chi tiết |
|------|--------|----------|
| **Chuẩn hóa FPS** | Resample về 30fps | Video 60fps → lấy cách frame, 24fps → lặp frame |
| **Chuẩn hóa Size** | Resize crop → 64×64 | Dùng bilinear interpolation |
| **Chuẩn hóa Pixel** | [0,255] → [0.0, 1.0] | float32 normalization |
| **Sliding Window** | 30 frame/cửa sổ, step=5 | Data Augmentation: tạo ~21K+ windows từ ~400 video |

**Tối ưu bộ nhớ**: Lưu track crops ra disk dưới dạng `.npy`, dùng **Lazy Dataset** + **memory-mapped numpy** để đọc on-the-fly khi training (không load hết vào RAM).

### 3. Kiến trúc Model

```
Input: (batch, 30, 3, 64, 64)
          │
          ▼
┌─────────────────────────────┐
│     Spatial CNN (4 blocks)  │  Trích xuất đặc trưng không gian
│     32 → 64 → 128 → 256    │  BatchNorm + ReLU + MaxPool
│     + SpatialDropout2d      │
│     + FC: 1024 → 256        │
└──────────┬──────────────────┘
           │  (batch, 30, 256)
           ▼
┌─────────────────────────────┐
│  Multi-scale Temporal Conv1D│  Trích xuất đặc trưng thời gian
│     Kernel 3 → 128 channels │  Cú đấm, giật (motion ngắn)
│     Kernel 5 → 64 channels  │  Xô đẩy (motion trung)
│     Kernel 7 → 64 channels  │  Đuổi nhau (motion dài)
│     Concat → 256, MaxPool   │
└──────────┬──────────────────┘
           │  (batch, 15, 256)
           ▼
┌─────────────────────────────┐
│    Bi-LSTM (2 layers)       │  Nắm bắt ngữ cảnh thời gian
│    Hidden: 128 × 2 = 256    │  Hai chiều: quá khứ + tương lai
│    + Dropout 0.3            │
└──────────┬──────────────────┘
           │  (batch, 15, 256)
           ▼
┌─────────────────────────────┐
│   Temporal Attention        │  Tập trung vào timestep quan trọng
│   Linear → Tanh → Linear   │
│   Softmax → Weighted Sum    │
└──────────┬──────────────────┘
           │  (batch, 256)
           ▼
┌─────────────────────────────┐
│      Classifier (FC)        │
│   256 → 128 → 64 → 2       │
│   BatchNorm + Dropout       │
└──────────┬──────────────────┘
           │
           ▼
    Violence / Non-Violence
```

### 4. Kỹ thuật huấn luyện

| Kỹ thuật | Chi tiết |
|----------|----------|
| **Optimizer** | AdamW (lr=3e-4, weight_decay=5e-4) |
| **LR Schedule** | Cosine Annealing Warm Restarts |
| **Loss** | CrossEntropy + Label Smoothing (0.2) + Class Weights |
| **MixUp** | alpha=0.3 — trộn 2 mẫu ngẫu nhiên để chống overfitting |
| **Early Stopping** | Patience = 20 epochs |
| **Gradient Clipping** | max_norm = 1.0 |
| **Ngưỡng phân loại** | 45% — nếu P(violence) ≥ 45% → phân loại là bạo lực |

### 5. Data Augmentation

| Augmentation | Xác suất | Mô tả |
|-------------|----------|--------|
| Horizontal Flip | 50% | Lật ngang cả sequence |
| Temporal Reverse | 50% | Phát ngược sequence |
| Random Speed | 30% | Tăng/giảm tốc 0.8x–1.2x |
| Brightness Jitter | 50% | Thay đổi độ sáng ±15% |
| Temporal Jitter | 30% | Xóa 1-2 frame, lặp frame kề |
| Gaussian Noise | 50% | Noise σ=0.015 |
| Random Erasing | 25% | Che 1 vùng nhỏ trong ảnh |

---

## 📊 Kết quả

### Dataset
- **Tổng**: ~400 video (200 violence + 200 non-violence)
- **Sau Sliding Window**: ~21,000+ cửa sổ 30-frame
- **Split**: 80% train / 20% val (stratified theo video)

### Hiệu năng
| Metric | Giá trị |
|--------|---------|
| **Val Accuracy** | ~89-92% |
| **Ngưỡng phân loại** | 45% |

---

## 📄 Mô tả các file

| File | Chức năng |
|------|-----------|
| `train_ai.py` | **Script chính** — Tiền xử lý dữ liệu (chuẩn hóa FPS, size, sliding window) + Huấn luyện model CNN-BiLSTM-Attention + Đánh giá + Xuất kết quả |
| `dectect_people.py` | Dùng YOLO + ByteTrack để phát hiện & tracking người trong video, xuất annotation XML chuẩn CVAT |
| `check_labels.py` | Kiểm tra chất lượng annotation bằng cách vẽ bounding box lên video gốc tạo preview |
| `rename.py` | Tiện ích đổi tên video theo quy chuẩn: `v_001.mp4`, `nv_001.mp4`,... |

---

## 🔧 Tiện ích

### Đổi tên video
```bash
python rename.py
# violence/xxx.mp4 → violence/v_001.mp4, v_002.mp4, ...
# non_violence/xxx.mp4 → non_violence/nv_001.mp4, nv_002.mp4, ...
```

### Kiểm tra annotation
```bash
# Sửa biến TARGET_XML trong check_labels.py
python check_labels.py
# → Xuất video preview vào fixed_previews/
```

---

## 📝 Ghi chú

- **Lần đầu chạy `train_ai.py`**: sẽ tiền xử lý toàn bộ video (~3-4 phút), kết quả cache vào `processed_tracks_v4/`. Các lần sau sẽ load cache ngay lập tức.
- **Chạy trên CPU**: Mỗi epoch mất ~5-10 phút. Khuyến nghị dùng GPU để tăng tốc.
- **Model đã train**: Lưu tại `best_model_v2.pth` (~8MB).

---

## 👤 Tác giả

Đồ án tốt nghiệp — Đại học

## 📜 License

Dự án phục vụ mục đích học tập và nghiên cứu.
