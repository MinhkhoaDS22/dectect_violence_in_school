# 🛡️ Detect Violence in School — SafeWatch

> **Đồ án tốt nghiệp HUTECH 2026** — Hệ thống **End-to-End** phát hiện hành vi bạo lực trong môi trường học đường sử dụng Deep Learning, tích hợp hoàn chỉnh từ Camera → AI → Cảnh báo tức thì.

📄 **[Báo cáo đồ án đầy đủ (PDF)](./report_DATN.pdf)**

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Demo End-to-End](#-demo-end-to-end)
- [Pipeline tổng thể](#-pipeline-tổng-thể)
- [Pipeline 1 — Huấn luyện](#-pipeline-1-huấn-luyện-mô-hình)
- [Kiến trúc mô hình AI](#-kiến-trúc-mô-hình-ai--cnn-bilstm-attention)
- [Pipeline 2 — Ứng dụng SafeWatch](#-pipeline-2-ứng-dụng-thực-tế--safewatch-inference)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Giao diện ứng dụng](#-giao-diện-ứng-dụng--safewatch)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Dataset](#-dataset)
- [Backend API](#-backend-api)
- [Cài đặt và chạy](#-cài-đặt-và-chạy)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## 🎯 Tổng quan

**SafeWatch** là hệ thống **End-to-End** phát hiện bạo lực học đường theo thời gian thực. Toàn bộ luồng từ đầu vào đến đầu ra được tích hợp liền mạch trong một hệ thống duy nhất:

| Thành phần | Công nghệ | Mô tả |
|---|---|---|
| **AI Model** | CNN-BiLSTM-Attention (~2.8M params) | Phân loại bạo lực từ chuỗi video 30 frame |
| **Object Tracking** | YOLO11s + ByteTrack | Phát hiện và theo dõi từng người trong video |
| **Backend** | FastAPI + Python | Xử lý video, chạy AI inference, quản lý cảnh báo |
| **Frontend** | Flutter Web | Giao diện upload, dashboard, lịch sử cảnh báo |
| **Alerting** | Gmail SMTP + Telegram Bot | Thông báo đa kênh ngay khi phát hiện |

---

## 🔗 Demo End-to-End

Toàn bộ hệ thống hoạt động theo một luồng **khép kín hoàn toàn tự động**. Người dùng chỉ cần upload video — phần còn lại hệ thống tự xử lý:

```
📹 Video từ camera/upload
        │
        ▼
[Flutter Web App]  ──→  Upload qua HTTP API
        │
        ▼
[FastAPI Backend]  ──→  Task Queue FIFO (xử lý nhiều camera đồng thời)
        │
        ▼
[YOLO11s + ByteTrack]  ──→  Detect + Track từng người (1 lần đọc video)
        │
        ▼
[CNN-BiLSTM-Attention]  ──→  Phân tích 30-frame window cho từng người
        │
        ▼
[Multi-level Decision]  ──→  Track-level + MAX Aggregation + Consecutive Check
        │
    ┌───┴───┐
    ▼       ▼
[BẠO LỰC]  [AN TOÀN]
    │           │
    ▼           ▼
✂️ Cắt clip   📊 Dashboard
📧 Email       cập nhật
📱 Telegram
🔊 Còi hú
```

**Điểm nổi bật End-to-End:**
- ✅ **Zero-touch**: Upload xong → hệ thống tự chạy, không cần thao tác thêm
- ✅ **Multi-camera**: Hàng đợi FIFO xử lý nhiều video tuần tự, không tràn RAM
- ✅ **Single-pass I/O**: Mỗi video chỉ được đọc **một lần duy nhất** — tối ưu hiệu suất
- ✅ **Real-time feedback**: Trạng thái Processing cập nhật ngay trên UI
- ✅ **Multi-channel alert**: Email + Telegram + Siren âm thanh đồng thời

---

## 🗺️ Pipeline Tổng Thể

Hệ thống gồm **hai luồng hoàn toàn tách biệt** — Training (offline) và Inference (online):

```mermaid
flowchart LR
    subgraph P1["🎓 PIPELINE 1 — HUẤN LUYỆN (Offline)"]
        direction TB
        D1["📹 389 Video thô\nViolence / Non-Violence"] --> T1["🏷️ Annotation\nCVAT + YOLO"]
        T1 --> T2["⚙️ Tiền xử lý\nCrop · Resize · ByteTrack · Sliding Window"]
        T2 --> T3["🧠 CNN-BiLSTM-Attention\nForward Pass + Backprop"]
        T3 --> T4["📉 Tối ưu hóa\nAdamW · Focal Loss · MixUp · Early Stopping"]
        T4 --> W["💾 best_model.pth\n~2.8M params · ~8MB"]
    end

    subgraph P2["🚀 PIPELINE 2 — ỨNG DỤNG (Online / SafeWatch)"]
        direction TB
        V1["📱 Flutter App\nUpload video"] --> V2["🔗 FastAPI Backend\nTask Queue FIFO"]
        V2 --> V3["👁️ YOLO11s + ByteTrack\nSingle-pass Tracking"]
        V3 --> V4["🪟 Sliding Window\nBatch Inference × 8"]
        V4 --> V5["⚖️ Multi-level Decision\nTrack-level + MAX + Consecutive"]
        V5 -->|"🚨 Bạo lực"| V6["✂️ Clip + 📧 Email\n📱 Telegram + 🔊 Siren"]
        V5 -->|"✅ An toàn"| V7["📊 Dashboard Update"]
        V6 --> V7
    end

    W -.->|"Load at startup"| V4
```

---

## 🎓 Pipeline 1: Huấn luyện Mô hình

```mermaid
flowchart TD
    A(["📁 INPUT\n389 video thô (.mp4)\nViolence / Non-Violence"]) --> B

    subgraph B["BƯỚC 1 — Thu thập & Gán nhãn (Annotation)"]
        B1["🎥 Video thô (.mp4)"]
        B2["🔲 CVAT + YOLO\nBounding box tự động từng người × từng frame"]
        B3["📄 Annotation XML\nTọa độ bounding box theo frame"]
        B1 --> B2 --> B3
    end

    B --> C

    subgraph C["BƯỚC 2 — Tiền xử lý Không gian–Thời gian"]
        C1["✂️ Crop từng người\ntheo bounding box CVAT"]
        C2["📐 Resize → 64×64 pixel\nChuẩn hóa [0, 1]"]
        C3["🔗 ByteTrack\nLiên kết cùng người xuyên suốt video"]
        C4["🪟 Sliding Window\nWindow=30 frame · Step=5 frame"]
        C5["📦 NumPy Array 4D\n(Frames, C, W, H) — ~2000+ windows"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    C --> D

    subgraph D["BƯỚC 3 — Forward Pass qua Mô hình"]
        D1["🖼️ Spatial CNN\nTrích đặc trưng không gian\n→ (B, 30, 256)"]
        D2["📈 Multi-scale Conv1D\nk=3·5·7 song song\n→ (B, 15, 256)"]
        D3["🔄 Bi-LSTM × 2 lớp\nNgữ cảnh 2 chiều\n→ (B, 15, 256)"]
        D4["🎯 Temporal Attention\nTập trung frame quan trọng\n→ (B, 256)"]
        D5["📊 FC Classifier\n256→128→64→2 → Softmax\n→ P(Violence) ∈ [0,1]"]
        D1 --> D2 --> D3 --> D4 --> D5
    end

    D --> E

    subgraph E["BƯỚC 4 — Tối ưu hóa & Huấn luyện"]
        E1["📐 Focal Loss γ=2.0\n+ Label Smoothing 0.1"]
        E2["⚡ AdamW lr=2e-4\n+ Cosine Annealing"]
        E3["🔀 MixUp α=0.2\n+ 7 kỹ thuật Augmentation"]
        E4["🛑 Early Stopping\nPatience=25 · Grad Clip=1.0"]
        E1 --- E2 --- E3 --- E4
    end

    E --> F(["💾 OUTPUT: best_model.pth\nTest Accuracy: 90% (Full Pipeline)\n~2.8M params · ~8MB"])
```

---

## 🧠 Kiến trúc Mô hình AI — CNN-BiLSTM-Attention

```mermaid
flowchart TD
    IN(["📥 Input: (Batch, 30, 3, 64, 64)\n30 frame crop của 1 người ≈ 1 giây video"])

    subgraph CNN["🖼️ SPATIAL CNN — Trích xuất đặc trưng không gian"]
        CN1["Block 1: Conv2D 3→32 + BN + ReLU + MaxPool  ·  64×64 → 32×32"]
        CN2["Block 2: Conv2D 32→64 + BN + ReLU + MaxPool  ·  32×32 → 16×16"]
        CN3["Block 3: Conv2D 64→128 + BN + ReLU + MaxPool  ·  16×16 → 8×8"]
        CN4["Block 4: Conv2D 128→256 + AdaptiveAvgPool(2×2) + SpatialDropout(0.15)"]
        CN5["FC: 1024 → 256 + Dropout(0.4)   →   Output: (B, 30, 256)"]
        CN1 --> CN2 --> CN3 --> CN4 --> CN5
    end

    subgraph MS["📈 MULTI-SCALE CONV1D — Đặc trưng chuyển động đa tốc độ"]
        MS1["Conv1D k=3 → 128ch  ·  ~100ms  ·  Cú đấm, giật tay đột ngột"]
        MS2["Conv1D k=5 → 64ch   ·  ~167ms  ·  Xô đẩy, ngã, ngăn cản"]
        MS3["Conv1D k=7 → 64ch   ·  ~233ms  ·  Đuổi nhau, hành hung kéo dài"]
        MS4["Concat (256ch) + MaxPool1D(2)   →   Output: (B, 15, 256)"]
        MS1 & MS2 & MS3 --> MS4
    end

    subgraph BI["🔄 BI-LSTM — Hiểu ngữ cảnh 2 chiều"]
        BL1["→ Forward LSTM: frame 1→15  (hidden=128)"]
        BL2["← Backward LSTM: frame 15→1  (hidden=128)"]
        BL3["Concat 128+128=256  +  Dropout(0.3)   →   Output: (B, 15, 256)"]
        BL1 & BL2 --> BL3
    end

    subgraph AT["🎯 TEMPORAL ATTENTION — Tập trung khoảnh khắc quan trọng"]
        AT1["Linear(256→64) + Tanh → điểm quan trọng eₜ"]
        AT2["Softmax(eₜ) → trọng số αₜ  (Σα = 1.0)"]
        AT3["Context = Σ(αₜ × hₜ)   →   Output: (B, 256)"]
        AT1 --> AT2 --> AT3
    end

    subgraph FC["📊 FC CLASSIFIER — Ra quyết định"]
        FC1["Linear 256→128 + BN + ReLU + Dropout(0.4)"]
        FC2["Linear 128→64 + ReLU + Dropout(0.3)"]
        FC3["Linear 64→2 → Softmax   →   [P(Non-Violence), P(Violence)]"]
        FC4{{"P(Violence) ≥ 0.45?"}}
        FC1 --> FC2 --> FC3 --> FC4
    end

    IN --> CNN --> MS --> BI --> AT --> FC
    FC4 -->|"✅ YES"| OUT1(["🚨 BẠO LỰC"])
    FC4 -->|"❌ NO"| OUT2(["✅ AN TOÀN"])
```

**Thông số mô hình:**

| Thông số | Giá trị |
|---|---|
| Tổng tham số | ~2,800,000 (~2.8M) — 100% trainable |
| Kích thước file | ~8 MB (float32) |
| Input | `(Batch, 30, 3, 64, 64)` |
| Output | `P(Violence) ∈ [0, 1]` |
| Ngưỡng quyết định | ≥ 45% → Bạo lực |

---

## 🚀 Pipeline 2: Ứng dụng thực tế — SafeWatch Inference

```mermaid
flowchart TD
    U(["👤 Người dùng — Flutter Web App\nUpload 1 hoặc nhiều video"]) --> S1

    subgraph S1["BƯỚC 1 — Tiếp nhận & Hàng chờ (Ingestion & Queue)"]
        S1A["📤 Upload video qua giao diện Web"]
        S1B["🔗 HTTP POST → FastAPI /api/analyze"]
        S1C["📋 Task Queue FIFO\nXử lý tuần tự — không tràn RAM/VRAM\nGiám sát nhiều camera đồng thời"]
        S1D["⏳ Trạng thái 'Processing' realtime trên UI"]
        S1A --> S1B --> S1C --> S1D
    end

    S1 --> S2

    subgraph S2["BƯỚC 2 — Single-pass Tracking (Tối ưu I/O)"]
        S2A["🎥 Đọc video CHỈ MỘT LẦN từ đầu đến cuối"]
        S2B["👁️ YOLO11s: Detect người\nconf=0.38 · imgsz=320\nChạy mỗi 2 frame → tiết kiệm ~50% thời gian CPU"]
        S2C["🔗 ByteTrack: Liên kết Track ID\nổn định xuyên suốt video"]
        S2D["✂️ Crop + Normalize 64×64 → RAM\nMỗi Track = danh sách ảnh của 1 người"]
        S2A --> S2B --> S2C --> S2D
    end

    S2 --> S3

    subgraph S3["BƯỚC 3 — Sliding Window & Batch Inference"]
        S3A["🔍 Lọc Track < 30 frame → bỏ qua"]
        S3B["🪟 Cắt Sliding Windows\n30 frame/window · Step=5 frame"]
        S3C["📦 Nhóm Batch (size=8)\nTận dụng song song hóa GPU/CPU"]
        S3D["🧠 CNN-BiLSTM-Attention\n(best_model.pth — load 1 lần khi khởi động)"]
        S3E["📊 Violence Score P(Violence)\ncho TỪNG window của TỪNG người"]
        S3A --> S3B --> S3C --> S3D --> S3E
    end

    S3 --> S4

    subgraph S4["BƯỚC 4 — Multi-level Decision Logic (Chống báo động giả)"]
        S4A["🔵 Tầng 1 — Track-level\nTrung bình tất cả windows của 1 Track\n≥ 45% → Mark track là violent"]
        S4B["🔴 Tầng 2 — Time-slot MAX Aggregation\nMỗi time-slot lấy MAX score của mọi người\n→ 1 người bạo lực vẫn phát hiện được\ndù đám đông người xem làm loãng điểm"]
        S4C["⏱️ Tầng 3 — Consecutive Check\n≥ 5 time-slot liên tiếp ≥ 45%\n→ ~2.5 giây liên tục\n→ Loại bỏ false positive từ cử chỉ ngẫu nhiên"]
        S4D{{"🚨 Is Violence?"}}
        S4A --> S4B --> S4C --> S4D
    end

    S4 --> S5

    subgraph S5["BƯỚC 5 — Post-processing & Alerting"]
        S5A["✂️ Video Clipper\nTrích xuất đúng đoạn bạo lực\n+ Watermark 'VIOLENCE DETECTED'"]
        S5B["📧 Gmail SMTP\nEmail HTML đẹp + clip đính kèm"]
        S5C["📱 Telegram Bot\nTin nhắn tức thì + video clip"]
        S5D["🔊 Siren Sound\nÂm thanh còi hú trên Web"]
        S5E["📊 Flutter Dashboard\nCập nhật timeline · Violence ratio\nMax persons · Số đoạn bạo lực"]
        S5A --> S5B & S5C & S5D & S5E
    end

    S4D -->|"✅ AN TOÀN"| SAFE(["✅ Dashboard cập nhật — Không cảnh báo"])
    S4D -->|"🚨 BẠO LỰC"| S5
```

### Tại sao dùng MAX thay vì MEAN?

> **Ví dụ thực tế**: 2 người đánh nhau `P=0.85` + 15 người đứng xem `P=0.08`
> - **MEAN** = `(2×0.85 + 15×0.08) / 17` = **0.17** → ❌ **Bỏ sót!**
> - **MAX** = `0.85` → ✅ **Phát hiện đúng!**

---

## 🏗️ Kiến trúc Hệ thống

```mermaid
flowchart TB
    subgraph USER["👤 Người dùng / Camera"]
        CAM["📹 Video (MP4, AVI, MOV, MKV)"]
    end

    subgraph FE["📱 Flutter Web App (SafeWatch)"]
        FE1["🏠 Dashboard\nThống kê & biểu đồ"]
        FE2["📹 Monitor\nUpload & Phân tích"]
        FE3["📋 History\nLịch sử cảnh báo + Timeline"]
        FE4["📋 Queue\nHàng đợi xử lý"]
        FE5["⚙️ Settings\nServer URL · Email · Telegram"]
    end

    subgraph BE["⚙️ FastAPI Backend"]
        API1["POST /api/analyze"]
        API2["GET /api/alerts"]
        API3["GET /api/clips"]
        API4["GET /api/health"]

        subgraph DET["🔍 Violence Detector"]
            PP1["YOLO11s\nPerson Detection"]
            PP2["ByteTrack\nPerson Tracking"]
            PP3["Crop + Normalize\n64×64 px"]
            PP4["CNN-BiLSTM-Attention\nWindow Inference"]
            PP5["MAX Aggregation\n+ Consecutive Check"]
            PP1 --> PP2 --> PP3 --> PP4 --> PP5
        end

        subgraph SVC["📨 Services"]
            SV1["✂️ Video Clipper\nCắt đoạn bạo lực"]
            SV2["📧 Gmail Notifier\nSMTP + HTML"]
            SV3["📱 Telegram Bot\nAPI send message"]
            SV4["💾 Alert Storage\nJSON CRUD"]
        end
    end

    subgraph AI["🤖 AI Models (Loaded at Startup)"]
        M1["💾 best_model.pth\nCNN-BiLSTM-Attention\n~2.8M params · ~8MB"]
        M2["👁️ yolo11s.pt\nUltralytics YOLO v11s\nPerson Detection"]
    end

    subgraph NOTIF["📬 Kênh thông báo"]
        N1["📧 Gmail"]
        N2["📱 Telegram"]
        N3["🔊 Browser Siren"]
    end

    CAM --> FE2
    FE -->|"HTTP (Dio)"| BE
    BE --> NOTIF
    DET --> SVC
    M1 -.->|"Singleton"| PP4
    M2 -.->|"Singleton"| PP1
    SV2 --> N1
    SV3 --> N2
    FE5 --> N3
```

---

## 📱 Giao diện Ứng dụng — SafeWatch

### Monitor — Upload & Phân tích

![Màn hình Monitor — Upload video để phân tích](./results/menu.png)

### Hàng chờ — Xử lý đa video (Multi-camera)

![Hàng chờ phân tích — hiển thị kết quả từng video](./results/hang_cho.png)

### Dashboard — Thống kê tổng quan

![Dashboard thống kê biểu đồ](./results/dashboard.png)

### Lịch sử — Violence Timeline

![Lịch sử cảnh báo với timeline đoạn bạo lực](./results/history.png)

### Cài đặt — Cấu hình Email / Telegram / Server

![Cài đặt kết nối và thông báo](./results/setting.png)

---

## 📊 Kết quả Thực nghiệm

### So sánh 3 phương pháp đánh giá (40 video test)

![So sánh 3 hướng đánh giá — Accuracy, F1, Precision, Recall](./test_result/comparison_summary.png)

| Phương pháp | Accuracy | F1 (Violence) | Precision | Recall |
|---|---|---|---|---|
| **H1: Sliding Window** | 77.5% | 80.8% | 70.4% | **95.0%** |
| **H2: Full App Pipeline** ⭐ | **90.0%** | **90.0%** | **90.0%** | 90.0% |
| **H3: No Sliding** | 79.3% | 83.4% | 76.5% | 91.8% |

> **H2 — Full App Pipeline** đạt hiệu suất tốt nhất **90% Accuracy** trên tập test 40 video. Đây là cách hệ thống thực sự hoạt động trong ứng dụng SafeWatch.

### Confusion Matrix — Test Set (Window-level)

![Confusion Matrix trên tập test](./results/confusion_matrix_test.png)

| Metric | Giá trị |
|---|---|
| True Positive (Violence đúng) | **1,154** |
| True Negative (Non-Violence đúng) | **1,725** |
| False Positive | 203 |
| False Negative | 143 |
| **Accuracy tổng** | **~89%** |

### Quá trình Huấn luyện — Loss & Accuracy

![Biểu đồ Loss và Accuracy theo Epoch](./results/improved_history.png)

- **Train Accuracy**: ~91% sau ~80 epochs
- **Val Accuracy**: ~90% — không có dấu hiệu overfitting nghiêm trọng
- **Hội tụ**: Loss giảm ổn định từ 0.61 → 0.42

### Phân phối Xác suất — Test Set

![Phân phối xác suất Violence trên tập test](./results/prob_dist_test.png)

Biểu đồ cho thấy mô hình **phân biệt rõ ràng** hai lớp:
- **Non-Violence** (xanh): tập trung ở vùng P < 0.35
- **Violence** (cam): tập trung ở vùng P > 0.70
- **Ngưỡng 45%** (đường đứt nét) tạo vùng phân tách hiệu quả

### Phân phối Dữ liệu Huấn luyện

![Phân phối Tracks theo tập Train/Val/Test](./results/data_distribution.png)

- **Tổng Tracks**: ~1,356 (Train: 919, Val: 300, Test: 137)
- **Tỉ lệ Violence**: ~36% trên mọi tập — phân phối đều, không bị lệch

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
│   └── non_violence/          # Video không có bạo lực
└── fix_labels/                # Annotation XML (CVAT format)
    ├── violence/
    └── non_violence/
```

### Chia dữ liệu

| Tập | Tỷ lệ | Video | Tracks (ước tính) |
|---|---|---|---|
| **Train** | 70% | ~210 | ~919 |
| **Validation** | 20% | ~60 | ~300 |
| **Test** | 10% | ~30 | ~137 |

> ⚠️ Chia theo **video** (không theo track) để tránh data leakage.

---

## ⚡ Backend API

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/api/health` | Kiểm tra trạng thái server |
| `POST` | `/api/analyze` | Upload video + phân tích bạo lực |
| `GET` | `/api/alerts` | Lấy danh sách cảnh báo |
| `GET` | `/api/alerts/{id}` | Xem chi tiết 1 cảnh báo |
| `DELETE` | `/api/alerts/{id}` | Xoá cảnh báo |
| `GET` | `/api/clips/{job_id}/{filename}` | Download clip bạo lực |

**Response mẫu từ `/api/analyze`:**
```json
{
  "is_violence": true,
  "segments": [
    { "start_sec": 2.5, "end_sec": 8.1, "confidence": 85.4 }
  ],
  "video_duration": 12.3,
  "violence_ratio": 0.62,
  "max_violent_persons": 3,
  "summary": "Phát hiện bạo lực (1 đoạn, 38/61 time-slot vượt ngưỡng, tối đa 3 người bạo lực cùng lúc)"
}
```

---

## 🚀 Cài đặt và chạy

### Yêu cầu

- Python 3.10+
- Flutter 3.10+ (SDK ^3.10.1)
- CUDA (khuyến nghị, để chạy GPU)

### 1. Huấn luyện mô hình

```bash
pip install torch torchvision opencv-python numpy matplotlib seaborn scikit-learn tqdm ultralytics

# Chuẩn bị dữ liệu
# - data/violence/ và data/non_violence/
# - fix_labels/violence/ và fix_labels/non_violence/

python train_ai.py
# → Output: results/best_model.pth
```

### 2. Chạy Backend

```bash
cd backend
pip install -r requirements.txt
pip install torch torchvision ultralytics

cp .env.example .env
# Điền Gmail App Password, Telegram Bot Token, đường dẫn model

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Cấu hình `.env`:**
```env
GMAIL_SENDER=your_email@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
TELEGRAM_BOT_TOKEN=your_bot_token_here
MODEL_PATH=../results/best_model.pth
YOLO_PATH=../yolo11s.pt
```

### 3. Chạy Flutter App

```bash
cd violence_app
flutter pub get
flutter run -d chrome
```

> Vào **Settings** để cấu hình Server URL → `http://localhost:8000`

---

## 📁 Cấu trúc thư mục

```
DATN/
├── train_ai.py              # Script huấn luyện CNN-BiLSTM-Attention
├── evaluate_model.py        # Đánh giá hiệu suất (3 hướng)
├── dectect_people.py        # Script detect người (testing)
├── report_DATN.pdf          # Báo cáo đồ án tốt nghiệp
│
├── data/                    # Video gốc
│   ├── violence/
│   └── non_violence/
├── fix_labels/              # Annotation XML (CVAT)
├── processed_tracks_v5/     # Cache tiền xử lý (.npy)
│
├── results/                 # Kết quả huấn luyện & ảnh app
│   ├── best_model.pth       # Model tốt nhất (~8MB)
│   ├── improved_history.png # Loss & Accuracy curves
│   ├── confusion_matrix_*.png
│   ├── prob_dist_*.png
│   ├── data_distribution.png
│   └── *.png                # Screenshots app SafeWatch
│
├── test_result/             # Kết quả đánh giá chi tiết
│   ├── comparison_summary.png
│   ├── evaluation_report.txt
│   └── approach*_*.png
│
├── backend/                 # FastAPI Backend
│   ├── main.py              # API endpoints
│   ├── violence_detector.py # Detection pipeline
│   ├── video_clipper.py     # Cắt clip bạo lực
│   ├── notifier.py          # Gmail + Telegram
│   └── requirements.txt
│
├── violence_app/            # Flutter App (SafeWatch)
│   └── lib/
│       ├── screens/         # 7 màn hình UI
│       ├── services/        # API, PDF, Queue, Sound
│       ├── models/          # AlertModel, QueueJob
│       └── widgets/         # Timeline, SoundBar
│
├── yolo11s.pt               # YOLO v11s weights
└── pipeline.txt             # Mô tả pipeline (text)
```

---

## 🛠️ Công nghệ sử dụng

### AI / Deep Learning

| Công nghệ | Phiên bản | Mục đích |
|---|---|---|
| PyTorch | - | Framework deep learning chính |
| Ultralytics YOLO | v11s | Person detection + ByteTrack |
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
| python-telegram-bot | 21.3 | Cảnh báo Telegram |

### Frontend

| Công nghệ | Phiên bản | Mục đích |
|---|---|---|
| Flutter | SDK ^3.10.1 | Framework UI cross-platform |
| Dio | 5.4.3 | HTTP client |
| fl_chart | 1.2.0 | Biểu đồ thống kê |
| flutter_animate | 4.5.0 | Micro-animations |
| pdf + printing | - | Xuất báo cáo PDF |

---

## 👨‍💻 Tác giả - Trương Minh Khoa

Đồ án tốt nghiệp — Trường Đại học Công nghệ TP.HCM (HUTECH) 2026

---

*SafeWatch — Phát hiện bạo lực, bảo vệ học đường* 🛡️
