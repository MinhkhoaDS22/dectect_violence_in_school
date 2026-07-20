# 🛡️ Detect Violence in School — SafeWatch

> **Đồ án tốt nghiệp HUTECH 2026** — Hệ thống phát hiện hành vi bạo lực trong môi trường học đường sử dụng Deep Learning, với ứng dụng Flutter và backend FastAPI.

📄 **[Báo cáo đồ án đầy đủ (PDF)](./report_DATN.pdf)**

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Pipeline tổng thể](#-pipeline-tổng-thể)
- [Pipeline Huấn luyện](#-pipeline-1-huấn-luyện-mô-hình)
- [Kiến trúc mô hình AI](#-kiến-trúc-mô-hình-ai--cnn-bilstm-attention)
- [Pipeline Ứng dụng](#-pipeline-2-ứng-dụng-thực-tế--safewatch-inference)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Dataset](#-dataset)
- [Backend API](#-backend-api)
- [Ứng dụng Flutter — SafeWatch](#-ứng-dụng-flutter--safewatch)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Cài đặt và chạy](#-cài-đặt-và-chạy)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## 🎯 Tổng quan

**SafeWatch** là hệ thống phát hiện bạo lực trong trường học theo thời gian thực, bao gồm 3 thành phần chính:

| Thành phần | Mô tả |
|---|---|
| **AI Model** | Mô hình CNN-BiLSTM-Attention (~2.8M params) huấn luyện trên dữ liệu video bạo lực/không bạo lực |
| **Backend** | FastAPI server xử lý video upload, chạy detection, cắt clip, gửi cảnh báo |
| **Flutter App** | Ứng dụng web SafeWatch — upload video, xem kết quả, quản lý cảnh báo |

---

## 🗺️ Pipeline Tổng Thể

Hệ thống được chia làm **hai luồng hoàn toàn tách biệt**:

```mermaid
flowchart LR
    subgraph P1["🎓 PIPELINE 1 — HUẤN LUYỆN (Offline)"]
        direction TB
        D1["📹 389 Video thô\n(Violence / Non-Violence)"] --> T1["🏷️ Annotation\nCVAT + YOLO"]
        T1 --> T2["⚙️ Tiền xử lý\nCrop · Resize · ByteTrack · Sliding Window"]
        T2 --> T3["🧠 Mô hình\nCNN-BiLSTM-Attention"]
        T3 --> T4["📉 Tối ưu hóa\nAdamW · Focal Loss · MixUp"]
        T4 --> W["💾 best_model.pth"]
    end

    subgraph P2["🚀 PIPELINE 2 — ỨNG DỤNG (Online / SafeWatch)"]
        direction TB
        V1["📱 Flutter App\nUpload video"] --> V2["🔗 FastAPI Backend\nTask Queue FIFO"]
        V2 --> V3["👁️ YOLO11s + ByteTrack\nSingle-pass Tracking"]
        V3 --> V4["🪟 Sliding Window\nBatch Inference × 8"]
        V4 --> V5["⚖️ Multi-level Decision\nTrack-level + MAX Aggregation"]
        V5 -->|"🚨 Bạo lực"| V6["✂️ Cắt clip\n📧 Email · 📱 Telegram"]
        V5 -->|"✅ An toàn"| V7["📊 Dashboard\nKhông cảnh báo"]
        V6 --> V7
    end

    W -.->|"Load model"| V4
```

---

## 🎓 Pipeline 1: Huấn luyện Mô hình

```mermaid
flowchart TD
    A(["📁 INPUT\n389 video thô\n.mp4 — Violence / Non-Violence"]) --> B

    subgraph B["BƯỚC 1 — Thu thập & Gán nhãn"]
        B1["🎥 Video thô (.mp4)"]
        B2["🔲 CVAT Annotation Tool\n+ YOLO hỗ trợ tự động"]
        B3["📄 XML files\nBounding box từng người × từng frame"]
        B1 --> B2 --> B3
    end

    B --> C

    subgraph C["BƯỚC 2 — Tiền xử lý Không-Thời gian"]
        C1["✂️ Crop từng người\ntheo bounding box"]
        C2["📐 Resize về\n64×64 pixel"]
        C3["🔗 ByteTrack\nLiên kết người xuyên suốt video"]
        C4["🪟 Sliding Window\nWindow=30 frame · Step=5 frame"]
        C5["📦 NumPy Array 4D\n(Frames, C, W, H) — Normalized [0,1]"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    C --> D

    subgraph D["BƯỚC 3 — Kiến trúc Mô hình (Forward Pass)"]
        D1["🖼️ Spatial CNN\nTrích xuất đặc trưng không gian\nOutput: (B, 30, 256)"]
        D2["📈 Multi-scale Conv1D\nk=3 · k=5 · k=7 song song\nOutput: (B, 15, 256)"]
        D3["🔄 Bi-LSTM × 2 lớp\nPhân tích chuỗi 2 chiều\nOutput: (B, 15, 256)"]
        D4["🎯 Temporal Attention\nTập trung frame quan trọng\nOutput: (B, 256)"]
        D5["📊 FC Classifier\n256→128→64→2 → Softmax\nOutput: P(Violence) ∈ [0,1]"]
        D1 --> D2 --> D3 --> D4 --> D5
    end

    D --> E

    subgraph E["BƯỚC 4 — Tối ưu hóa & Huấn luyện"]
        E1["📐 Focal Loss\nγ=2.0 + Label Smoothing 0.1"]
        E2["⚡ AdamW\nlr=2e-4 · Cosine Annealing"]
        E3["🔀 MixUp α=0.2\n+ 7 Data Augmentation"]
        E4["🛑 Early Stopping\nPatience=25 · Gradient Clip=1.0"]
        E1 --- E2 --- E3 --- E4
    end

    E --> F(["💾 OUTPUT\nbest_model.pth\nTest Accuracy ~89%"])
```

---

## 🧠 Kiến trúc Mô hình AI — CNN-BiLSTM-Attention

```mermaid
flowchart TD
    IN(["📥 Input\n(Batch, 30 frames, 3, 64, 64)\n30 frame ảnh crop của 1 người ~ 1 giây video"])

    subgraph CNN["🖼️ SPATIAL CNN — Trích xuất đặc trưng không gian"]
        CN1["Block 1: Conv2D 3→32 + BN + ReLU + MaxPool\n96×96 → 48×48"]
        CN2["Block 2: Conv2D 32→64 + BN + ReLU + MaxPool\n48×48 → 24×24"]
        CN3["Block 3: Conv2D 64→128 + BN + ReLU + MaxPool\n24×24 → 12×12"]
        CN4["Block 4: Conv2D 128→256 + AdaptiveAvgPool → 2×2\n+ Spatial Dropout(0.15)"]
        CN5["FC: 1024 → 256 + Dropout(0.4)"]
        CN1 --> CN2 --> CN3 --> CN4 --> CN5
    end

    subgraph MS["📈 MULTI-SCALE CONV1D — Đặc trưng chuyển động"]
        MS1["Conv1D k=3 → 128ch\n~100ms: cú đấm, giật tay"]
        MS2["Conv1D k=5 → 64ch\n~167ms: xô đẩy, ngã"]
        MS3["Conv1D k=7 → 64ch\n~233ms: đuổi nhau, hành hung"]
        MS4["Concat (128+64+64=256ch)\n+ MaxPool1D(2)\n30 timestep → 15 timestep"]
        MS1 & MS2 & MS3 --> MS4
    end

    subgraph BI["🔄 BI-LSTM — Hiểu ngữ cảnh 2 chiều"]
        BL1["Forward LSTM\nframe 1→2→...→15\n(hidden=128)"]
        BL2["Backward LSTM\nframe 15→14→...→1\n(hidden=128)"]
        BL3["Concat: 128+128=256\nDropout(0.3) giữa 2 lớp\nOutput: (B, 15, 256)"]
        BL1 & BL2 --> BL3
    end

    subgraph AT["🎯 TEMPORAL ATTENTION — Chú ý khoảnh khắc quan trọng"]
        AT1["Linear(256→64) + Tanh\n→ điểm quan trọng e_t"]
        AT2["Softmax(e_t)\n→ trọng số α_t, Σα=1"]
        AT3["Context = Σ(αₜ × hₜ)\nOutput: (B, 256)"]
        AT1 --> AT2 --> AT3
    end

    subgraph FC["📊 FC CLASSIFIER — Kết luận"]
        FC1["Linear 256→128 + BN + ReLU + Dropout(0.4)"]
        FC2["Linear 128→64 + ReLU + Dropout(0.3)"]
        FC3["Linear 64→2 → Softmax"]
        FC4{{"P(Violence) ≥ 0.45?"}}
        FC1 --> FC2 --> FC3 --> FC4
    end

    IN --> CNN --> MS --> BI --> AT --> FC
    FC4 -->|"✅ Có"| OUT1(["🚨 BẠO LỰC"])
    FC4 -->|"❌ Không"| OUT2(["✅ AN TOÀN"])
```

### Thông số mô hình

| Thành phần | Chi tiết |
|---|---|
| **Tổng tham số** | ~2,800,000 (~2.8M) — 100% trainable |
| **Kích thước file** | ~8 MB (float32) |
| **Input** | (Batch, 30, 3, 64, 64) |
| **Output** | P(Violence) ∈ [0, 1] |
| **Ngưỡng quyết định** | ≥ 45% → Bạo lực |

### Kỹ thuật huấn luyện

| Kỹ thuật | Chi tiết |
|---|---|
| **Loss** | Focal Loss (γ=2.0) + Class Weights + Label Smoothing (0.1) |
| **Optimizer** | AdamW (lr=2e-4, weight_decay=3e-4) |
| **LR Schedule** | 5-epoch Warmup → Cosine Annealing Warm Restarts |
| **MixUp** | α=0.2 — trộn 2 mẫu ngẫu nhiên, giảm overfitting |
| **Augmentation** | Flip ngang · Temporal Reverse · Speed Variation · Brightness Jitter · Gaussian Noise · Random Erasing (7 kỹ thuật) |
| **Early Stopping** | Patience = 25 epoch |
| **Gradient Clipping** | max_norm = 1.0 |

---

## 🚀 Pipeline 2: Ứng dụng thực tế — SafeWatch Inference

```mermaid
flowchart TD
    U(["👤 Người dùng\nFlutter Web App"]) --> S1

    subgraph S1["BƯỚC 1 — Tiếp nhận & Hàng chờ"]
        S1A["📤 Upload video qua UI"]
        S1B["🔗 HTTP API (Dio → FastAPI)"]
        S1C["📋 Task Queue FIFO\nXử lý tuần tự, tránh tràn RAM/VRAM"]
        S1D["⏳ Trạng thái 'Processing'\nhiển thị realtime trên UI"]
        S1A --> S1B --> S1C --> S1D
    end

    S1 --> S2

    subgraph S2["BƯỚC 2 — Rút trích đối tượng (Single-pass Tracking)"]
        S2A["🎥 Load video\n(chỉ đọc MỘT LẦN — tối ưu I/O)"]
        S2B["👁️ YOLO11s\nPhát hiện người\nconf=0.38 · imgsz=320\nChạy mỗi 2 frame (~50% thời gian)"]
        S2C["🔗 ByteTrack\nLiên kết người xuyên suốt video\n(Track ID ổn định)"]
        S2D["✂️ Crop + Resize 64×64\nLưu vào RAM theo từng Track ID"]
        S2A --> S2B --> S2C --> S2D
    end

    S2 --> S3

    subgraph S3["BƯỚC 3 — Sliding Window & Batch Inference"]
        S3A["🔍 Lọc Track quá ngắn\n(< 30 frame → bỏ qua)"]
        S3B["🪟 Cắt Sliding Windows\n30 frame/window · Step=5 frame"]
        S3C["📦 Nhóm thành Batch (size=8)\nTăng tốc tính toán GPU/CPU"]
        S3D["🧠 CNN-BiLSTM-Attention\n(best_model.pth)"]
        S3E["📊 Violence Score\nP(Violence) mỗi window ∈ [0,1]"]
        S3A --> S3B --> S3C --> S3D --> S3E
    end

    S3 --> S4

    subgraph S4["BƯỚC 4 — Logic Kết luận Đa tầng"]
        S4A["🔵 Tầng 1: Track-level\nTrung bình điểm tất cả windows\ncủa 1 Track ≥ 45% → Mark violent"]
        S4B["🔴 Tầng 2: Time-slot MAX\nMỗi time-slot lấy MAX score\ncủa tất cả mọi người\n→ 1 người bạo lực vẫn phát hiện được"]
        S4C["⏱️ Tầng 3: Consecutive Check\n≥ 5 time-slot liên tiếp ≥ 45%\n→ ~2.5 giây — chống báo động giả"]
        S4D{{"🚨 Is Violence?"}}
        S4A --> S4B --> S4C --> S4D
    end

    S4 --> S5

    subgraph S5["BƯỚC 5 — Hậu xử lý & Cảnh báo"]
        direction LR
        S5A["✂️ Video Clipper\nTrích xuất đoạn bạo lực\n+ Watermark 'VIOLENCE DETECTED'"]
        S5B["📧 Gmail SMTP\nEmail HTML + clip đính kèm"]
        S5C["📱 Telegram Bot\nTin nhắn + video clip"]
        S5D["📊 Flutter Dashboard\nCập nhật violence timeline\n+ Âm thanh còi hú (Siren)"]
        S5A --> S5B & S5C & S5D
    end

    S4D -->|"✅ AN TOÀN"| SAFE(["✅ Dashboard cập nhật\nKhông cảnh báo"])
    S4D -->|"🚨 BẠO LỰC"| S5
```

### Logic phát hiện — Tại sao dùng MAX thay vì MEAN?

> **Ví dụ thực tế**: 2 người đánh nhau (P=0.85) + 15 người đứng xem (P=0.08)
> - **MEAN** = 0.17 → ❌ **Bỏ sót bạo lực!**
> - **MAX** = 0.85 → ✅ **Phát hiện đúng!**

Trong môi trường học đường, chỉ cần **1–2 người** có hành vi bạo lực giữa đám đông là đủ để cảnh báo. Nếu dùng MEAN, xác suất bạo lực sẽ bị "làm loãng" bởi đám đông người xem.

---

## 🏗️ Kiến trúc Hệ thống

```mermaid
flowchart TB
    subgraph FE["📱 Flutter Web App (SafeWatch)"]
        FE1["🏠 Dashboard\nThống kê & biểu đồ"]
        FE2["📹 Monitor\nUpload & Phân tích"]
        FE3["📋 History\nLịch sử cảnh báo"]
        FE4["⚙️ Settings\nEmail · Telegram · Server URL"]
    end

    subgraph BE["⚙️ FastAPI Backend (Python)"]
        BE1["/api/analyze\nUpload + Detect"]
        BE2["/api/alerts\nCRUD cảnh báo"]
        BE3["/api/clips\nDownload clip"]
        BE4["/api/health\nKiểm tra server"]

        subgraph PIPELINE["🔍 Violence Detector Pipeline"]
            PP1["YOLO11s Tracking"]
            PP2["Crop + Normalize"]
            PP3["CNN-BiLSTM-Attention"]
            PP4["Multi-level Aggregation"]
            PP1 --> PP2 --> PP3 --> PP4
        end

        subgraph SERVICES["📨 Services"]
            SV1["✂️ Video Clipper"]
            SV2["📧 Notifier\nGmail + Telegram"]
            SV3["💾 Alert Storage\nJSON-based CRUD"]
        end
    end

    subgraph MODELS["🤖 AI Models"]
        M1["🏋️ best_model.pth\nCNN-BiLSTM-Attention\n~2.8M params · ~8MB"]
        M2["👁️ yolo11s.pt\nUltralytics YOLO v11\nPerson Detection"]
    end

    FE -->|"HTTP (Dio)"| BE
    BE4 -.->|"Health check"| FE
    PIPELINE --> SERVICES
    M1 -.->|"Loaded at startup"| PP3
    M2 -.->|"Loaded at startup"| PP1
```

---

## ⚡ Backend API

Backend xây dựng bằng **FastAPI**, cung cấp các endpoint:

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/health` | Kiểm tra trạng thái server |
| `POST` | `/api/analyze` | Upload video + phân tích bạo lực |
| `GET` | `/api/alerts` | Lấy danh sách cảnh báo |
| `GET` | `/api/alerts/{id}` | Xem chi tiết 1 cảnh báo |
| `DELETE` | `/api/alerts/{id}` | Xoá cảnh báo |
| `GET` | `/api/clips/{job_id}/{filename}` | Download clip bạo lực |

### Tính năng

- **Phân tích video**: Upload → YOLO tracking → CNN-BiLSTM → trả kết quả JSON
- **Cắt clip tự động**: Cắt chính xác đoạn bạo lực với watermark "VIOLENCE DETECTED"
- **Thông báo đa kênh**:
  - 📧 **Gmail**: Email HTML đẹp kèm clip đính kèm (SMTP + App Password)
  - 📱 **Telegram Bot**: Gửi tin nhắn + video clip qua Bot API
- **Lưu trữ cảnh báo**: JSON-based storage, hỗ trợ CRUD

---

## 📱 Ứng dụng Flutter — SafeWatch

Ứng dụng Flutter Web với giao diện dark mode hiện đại.

### Các màn hình

| Màn hình | Mô tả |
|---|---|
| **Onboarding** | Giới thiệu ứng dụng lần đầu sử dụng |
| **Home** | Navigation bar với 4 tab chính |
| **Monitor** | Upload video + xem kết quả phân tích real-time |
| **Dashboard** | Thống kê tổng quan: biểu đồ tròn, bar chart, timeline |
| **History** | Lịch sử cảnh báo, tìm kiếm, chi tiết từng alert |
| **Queue** | Hàng đợi xử lý khi upload nhiều video |
| **Settings** | Cấu hình email, Telegram, server URL |

### Tính năng nổi bật

- 🎨 **Giao diện Dark Mode** premium với animations (flutter_animate)
- 📊 **Dashboard thống kê** với biểu đồ tương tác (fl_chart)
- 📄 **Xuất báo cáo PDF** chi tiết kết quả phân tích
- 🔔 **Âm thanh cảnh báo** tuỳ chỉnh được
- 📹 **Violence Timeline** — hiển thị timeline đoạn bạo lực trực quan
- ⚙️ **Cấu hình linh hoạt** — thay đổi server URL, email, Telegram ngay trong app

### Thư viện Flutter sử dụng

| Package | Mục đích |
|---|---|
| `dio` | HTTP client cho API calls |
| `file_picker` | Chọn file video để upload |
| `fl_chart` | Biểu đồ thống kê |
| `flutter_animate` | Micro-animations cho UI |
| `google_fonts` | Typography đẹp |
| `shared_preferences` | Lưu cài đặt local |
| `pdf` + `printing` | Xuất báo cáo PDF |
| `intl` | Format ngày giờ |
| `percent_indicator` | Thanh tiến trình |

---

## 📦 Dataset

### Tải xuống

> 📥 **Download dataset tại [GitHub Releases](https://github.com/MinhkhoaDS22/dectect_violence_in_school/releases)**

Dataset bao gồm file `data_labels.zip` chứa toàn bộ video gốc và annotation labels.

### Thông tin dataset

| Thông tin | Chi tiết |
|---|---|
| **Tổng số video** | 300 video |
| **Phân loại** | Violence (bạo lực) + Non-Violence (không bạo lực) |
| **Nguồn** | Thu thập từ môi trường trường học |
| **Annotation** | Bounding box theo frame cho từng người (CVAT format — XML) |
| **FPS gốc** | Đa dạng, được chuẩn hoá về 30 FPS khi tiền xử lý |

### Cấu trúc bên trong `data_labels.zip`

```
data_labels.zip
├── data/
│   ├── violence/              # Video có hành vi bạo lực
│   │   ├── v_001.mp4
│   │   └── ...
│   └── non_violence/          # Video không có bạo lực
│       ├── nv_001.mp4
│       └── ...
└── fix_labels/                # Annotation XML (CVAT format)
    ├── violence/
    │   ├── v_001.xml
    │   └── ...
    └── non_violence/
        ├── nv_001.xml
        └── ...
```

### Chia dữ liệu

Dữ liệu được chia theo **video** (không theo track) để tránh data leakage:

| Tập | Tỷ lệ | Mục đích |
|---|---|---|
| **Train** | 70% | Huấn luyện mô hình |
| **Validation** | 20% | Điều chỉnh hyperparameter, early stopping |
| **Test** | 10% | Đánh giá hiệu suất cuối cùng |

> ⚠️ **Lưu ý**: 300 video → ~500-800 tracks → ~2000+ sliding windows (mỗi window = 30 frame = 1 giây).

---

## 📊 Kết quả thực nghiệm

### Hiệu suất mô hình

| Metric | Giá trị |
|---|---|
| **Train Accuracy** | ~90% |
| **Test Accuracy** | ~89% |
| **Tổng tham số** | ~2.8M |
| **Kích thước model** | ~8 MB |
| **Ngưỡng phân loại** | 45% |
| **Consecutive Windows** | ≥ 5 (~2.5 giây liên tiếp) |

> **Lưu ý**: Confusion matrix đánh giá ở cấp **sliding window** (mỗi window = 1 giây video), không phải cấp video. 300 video → ~500-800 tracks → ~2000+ windows.

---

## 🚀 Cài đặt và chạy

### Yêu cầu

- Python 3.10+
- Flutter 3.10+ (SDK ^3.10.1)
- CUDA (khuyến nghị, để chạy GPU)

### 1. Huấn luyện mô hình

```bash
# Cài đặt dependencies
pip install torch torchvision opencv-python numpy matplotlib seaborn scikit-learn tqdm ultralytics

# Chuẩn bị dữ liệu
# - Đặt video vào data/violence/ và data/non_violence/
# - Đặt XML annotation (CVAT) vào fix_labels/violence/ và fix_labels/non_violence/

# Chạy huấn luyện
python train_ai.py
# → Output: best_model.pth, confusion matrix, biểu đồ
```

### 2. Chạy Backend

```bash
cd backend

# Cài đặt dependencies
pip install -r requirements.txt
pip install torch torchvision ultralytics

# Cấu hình
cp .env.example .env
# → Sửa file .env: điền Gmail App Password, Telegram Bot Token, đường dẫn model

# Chạy server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# hoặc
python main.py
```

**Cấu hình `.env`**:

```env
# Gmail (bật 2FA → tạo App Password)
GMAIL_SENDER=your_email@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# Telegram Bot (tạo bằng @BotFather)
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Đường dẫn model
MODEL_PATH=../results/best_model.pth
YOLO_PATH=../yolo11s.pt
```

### 3. Chạy Flutter App

```bash
cd violence_app

# Cài đặt dependencies
flutter pub get

# Chạy ứng dụng web
flutter run -d chrome

# Hoặc build web
flutter build web
```

> Sau khi chạy app, vào **Settings** để cấu hình Server URL trỏ về backend (mặc định `http://localhost:8000`).

---

## 📁 Cấu trúc thư mục

```
DATN/
├── train_ai.py              # Script huấn luyện model CNN-BiLSTM-Attention
├── evaluate_model.py        # Script đánh giá hiệu suất mô hình
├── dectect_people.py        # Script detect người (testing)
├── rename.py                # Đổi tên file tiện ích
├── report_DATN.pdf          # Báo cáo đồ án tốt nghiệp
│
├── data/                    # Dữ liệu video gốc
│   ├── violence/            #   Video bạo lực
│   └── non_violence/        #   Video không bạo lực
│
├── fix_labels/              # Annotation XML (CVAT format)
│   ├── violence/
│   └── non_violence/
│
├── processed_tracks_v5/     # Cache dữ liệu đã tiền xử lý (.npy)
│
├── results/                 # Kết quả huấn luyện
│   ├── best_model.pth       #   Model tốt nhất
│   ├── improved_history.png #   Biểu đồ Loss & Accuracy
│   ├── confusion_matrix_*.png
│   └── data_distribution.png
│
├── backend/                 # FastAPI Backend
│   ├── main.py              #   API endpoints
│   ├── violence_detector.py #   Detection pipeline
│   ├── video_clipper.py     #   Cắt clip bạo lực
│   ├── notifier.py          #   Gửi Gmail + Telegram
│   ├── requirements.txt     #   Python dependencies
│   ├── .env.example         #   Mẫu cấu hình
│   ├── uploads/             #   Video upload tạm
│   └── clips/               #   Clip bạo lực đã cắt
│
├── violence_app/            # Flutter App (SafeWatch)
│   ├── lib/
│   │   ├── main.dart        #   Entry point
│   │   ├── models/          #   Data models (AlertModel, QueueJob)
│   │   ├── screens/         #   7 màn hình UI
│   │   ├── services/        #   API, PDF, Queue, Sound services
│   │   ├── theme/           #   Dark theme configuration
│   │   └── widgets/         #   Custom widgets (Timeline, SoundBar)
│   ├── pubspec.yaml         #   Flutter dependencies
│   └── web/                 #   Web platform files
│
├── yolo11s.pt               # YOLOv11s pre-trained weights
└── pipeline.txt             # Mô tả chi tiết pipeline (text)
```

---

## 🛠️ Công nghệ sử dụng

### AI / Deep Learning

| Công nghệ | Phiên bản | Mục đích |
|---|---|---|
| PyTorch | - | Framework deep learning chính |
| Ultralytics YOLO | v11s | Phát hiện và tracking người (ByteTrack) |
| OpenCV | 4.9.0 | Xử lý video, crop, resize |
| scikit-learn | - | Metrics, confusion matrix |
| NumPy | 1.26.4 | Xử lý dữ liệu số |

### Backend

| Công nghệ | Phiên bản | Mục đích |
|---|---|---|
| FastAPI | 0.111.0 | REST API framework |
| Uvicorn | 0.30.0 | ASGI server |
| python-dotenv | 1.0.1 | Quản lý biến môi trường |
| smtplib | built-in | Gửi email Gmail |
| python-telegram-bot | 21.3 | Gửi cảnh báo Telegram |

### Frontend

| Công nghệ | Phiên bản | Mục đích |
|---|---|---|
| Flutter | SDK ^3.10.1 | Framework UI cross-platform |
| Dart | - | Ngôn ngữ lập trình |
| Dio | 5.4.3 | HTTP client |
| fl_chart | 1.2.0 | Biểu đồ thống kê |
| flutter_animate | 4.5.0 | Animations |

### Annotation

| Công cụ | Mục đích |
|---|---|
| CVAT | Gán nhãn bounding box theo frame cho từng người trong video |

---

## 👨‍💻 Tác giả

Đồ án tốt nghiệp — Trường Đại học Công nghệ TP.HCM (HUTECH) 2026

---

*SafeWatch — Phát hiện bạo lực, bảo vệ học đường* 🛡️
