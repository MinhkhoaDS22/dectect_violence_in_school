import os
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import random
import copy
from collections import defaultdict

# ==========================================
# 1. TIỀN XỬ LÝ: CHUẨN HÓA FRAME RATE, SIZE & SLIDING WINDOW
#    (Phiên bản tối ưu bộ nhớ — lưu track crops ra disk)
# ==========================================

# --- Hằng số chuẩn hóa ---
TARGET_FPS = 30       # FPS chuẩn cho mọi video
TARGET_IMG_SIZE = 64  # Kích thước ảnh chuẩn (pixels)
WINDOW_LENGTH = 30    # Số frame mỗi cửa sổ
WINDOW_STEP = 5       # Bước nhảy sliding window (Data Augmentation)


def normalize_frame_indices(original_fps, target_fps, total_frames):
    """Tính danh sách chỉ số frame cần giữ lại sau khi chuẩn hoá FPS.
    
    Ví dụ: Video gốc 60fps → target 30fps: lấy frame 0, 2, 4, 6, ...
           Video gốc 24fps → target 30fps: nội suy lấy frame 0, 0, 1, 2, 2, 3, ...
    
    Returns:
        List[int]: danh sách chỉ số frame gốc cần lấy (có thể trùng khi upsample).
    """
    if original_fps <= 0 or total_frames <= 0:
        return list(range(total_frames))
    
    duration = total_frames / original_fps          # Thời lượng video (giây)
    target_total = int(round(duration * target_fps)) # Số frame sau chuẩn hoá
    
    if target_total <= 0:
        return list(range(total_frames))
    
    # Ánh xạ tuyến tính: frame mới i → frame gốc gần nhất
    indices = []
    for i in range(target_total):
        orig_idx = int(round(i * (total_frames - 1) / max(target_total - 1, 1)))
        orig_idx = min(orig_idx, total_frames - 1)
        indices.append(orig_idx)
    
    return indices


def extract_and_save_tracks(video_path, xml_path, label, save_dir, video_id,
                             seq_length=WINDOW_LENGTH,
                             img_size=TARGET_IMG_SIZE,
                             target_fps=TARGET_FPS):
    """Trích xuất track crops từ video, chuẩn hoá và lưu ra disk.
    
    Pipeline tiền xử lý (giữ nguyên):
        1. Chuẩn hoá Frame Rate: resample về target_fps (mặc định 30fps).
        2. Chuẩn hoá Size: resize crop về (img_size × img_size) pixels.
        3. Chuẩn hoá giá trị pixel: normalize về [0, 1].
    
    Thay vì tạo sliding windows trong RAM, hàm này lưu mỗi track thành
    file .npy riêng biệt trên disk (chỉ ~vài MB/track).
    
    Returns:
        list[dict]: manifest entries, mỗi entry gồm:
            - "file": đường dẫn file .npy
            - "label": nhãn (0 hoặc 1)
            - "num_frames": số frame trong track
    """
    entries = []
    
    # --- Đọc annotation XML ---
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    tracks = {}
    for track in root.findall('track'):
        track_id = track.get('id')
        tracks[track_id] = []
        for box in track.findall('box'):
            if box.get('outside') == '1':
                continue
            frame_idx = int(box.get('frame'))
            x1 = float(box.get('xtl'))
            y1 = float(box.get('ytl'))
            x2 = float(box.get('xbr'))
            y2 = float(box.get('ybr'))
            tracks[track_id].append({'frame': frame_idx, 'box': (x1, y1, x2, y2)})
            
    valid_tracks = {tid: data for tid, data in tracks.items() if len(data) >= seq_length}
    if not valid_tracks:
        return entries

    # --- Bước 1: Đọc video & chuẩn hoá FPS ---
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tính chỉ số frame cần lấy sau chuẩn hoá FPS
    if original_fps > 0 and abs(original_fps - target_fps) > 0.5:
        resampled_indices = normalize_frame_indices(original_fps, target_fps, total_frames)
    else:
        resampled_indices = list(range(total_frames))
    
    # Đọc toàn bộ frame gốc cần thiết vào bộ nhớ
    needed_originals = sorted(set(resampled_indices))
    if not needed_originals:
        cap.release()
        return entries
    
    original_frames = {}
    frame_idx = 0
    max_needed = max(needed_originals)
    needed_set = set(needed_originals)  # set lookup nhanh hơn list
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in needed_set:
            original_frames[frame_idx] = frame
        frame_idx += 1
        if frame_idx > max_needed:
            break
    cap.release()
    
    # --- Bước 2 & 3: Crop, resize, normalize cho từng track ---
    track_crops = {tid: [] for tid in valid_tracks.keys()}
    
    for new_idx, orig_idx in enumerate(resampled_indices):
        frame = original_frames.get(orig_idx)
        if frame is None:
            continue
        
        for tid, data_list in valid_tracks.items():
            box_data = next((item for item in data_list if item["frame"] == orig_idx), None)
            
            if box_data:
                x1, y1, x2, y2 = box_data['box']
                h_img, w_img = frame.shape[:2]
                
                # Mở rộng bbox 10% để lấy thêm context xung quanh
                bw = x2 - x1
                bh = y2 - y1
                x1 = max(0, int(x1 - 0.05 * bw))
                y1 = max(0, int(y1 - 0.05 * bh))
                x2 = min(w_img, int(x2 + 0.05 * bw))
                y2 = min(h_img, int(y2 + 0.05 * bh))
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    # Bước 2: Chuẩn hoá kích thước → (img_size × img_size)
                    crop_resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                    # Bước 3: Chuẩn hoá pixel [0, 255] → [0.0, 1.0]
                    crop_norm = crop_rgb.astype(np.float32) / 255.0
                    # Chuyển (H, W, 3) → (3, H, W) rồi append
                    crop_chw = np.transpose(crop_norm, (2, 0, 1))
                    track_crops[tid].append(crop_chw)
    
    # Giải phóng frame gốc khỏi RAM
    del original_frames
    
    # --- Lưu mỗi track thành file .npy trên disk ---
    os.makedirs(save_dir, exist_ok=True)
    
    for tid, crops in track_crops.items():
        if len(crops) < seq_length:
            continue
        
        # Stack thành numpy array: (num_frames, 3, img_size, img_size)
        track_array = np.stack(crops, axis=0)
        
        # Lưu file
        track_file = os.path.join(save_dir, f"{video_id}_t{tid}.npy")
        np.save(track_file, track_array)
        
        entries.append({
            "file": track_file,
            "label": int(label),
            "num_frames": len(crops),
        })
    
    return entries


def prepare_dataset(data_dir, labels_dir, cache_dir="processed_tracks_v4",
                    img_size=TARGET_IMG_SIZE, target_fps=TARGET_FPS):
    """Tiền xử lý video → lưu track crops ra disk + tạo manifest.
    
    Pipeline (giữ nguyên):
        1. Chuẩn hoá FPS → target_fps (mặc định 30fps).
        2. Chuẩn hoá size → img_size × img_size pixels.
        3. Chuẩn hoá pixel → [0, 1].
    
    Kết quả:
        - Thư mục cache_dir/ chứa các file .npy (mỗi track ~vài MB).
        - File manifest.json liệt kê tất cả tracks + labels + num_frames.
    
    Returns:
        manifest: list[dict] — danh sách tracks đã xử lý.
    """
    manifest_path = os.path.join(cache_dir, "manifest.json")
    
    # Nếu đã xử lý rồi → load manifest
    if os.path.exists(manifest_path):
        print(f"📦 Đã tìm thấy dữ liệu đã tiền xử lý tại '{cache_dir}/'. Đang load manifest...")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Tính thống kê
        total_windows = 0
        for entry in manifest:
            n = entry["num_frames"]
            total_windows += max(0, (n - WINDOW_LENGTH) // WINDOW_STEP + 1)
        
        n0 = sum(1 for e in manifest if e["label"] == 0)
        n1 = sum(1 for e in manifest if e["label"] == 1)
        print(f"   → {len(manifest)} tracks ({n0} non-violence, {n1} violence)")
        print(f"   → ~{total_windows} sliding windows (length={WINDOW_LENGTH}, step={WINDOW_STEP})")
        print(f"   → Cấu hình: FPS={target_fps}, Size={img_size}×{img_size}")
        return manifest
    
    # Xử lý từ đầu
    os.makedirs(cache_dir, exist_ok=True)
    manifest = []
    
    categories = {'non_violence': 0, 'violence': 1}
    
    print("⏳ Bắt đầu trích xuất dữ liệu từ Video và XML...")
    print(f"   📐 Chuẩn hoá: FPS={target_fps}, Size={img_size}×{img_size}, "
          f"Window={WINDOW_LENGTH} frames, Step={WINDOW_STEP} frames")
    print(f"   💾 Lưu track crops vào: '{cache_dir}/'")
    
    for category, label in categories.items():
        vid_folder = os.path.join(data_dir, category)
        xml_folder = os.path.join(labels_dir, category)
        
        if not os.path.exists(vid_folder):
            continue
        
        # Tạo subfolder cho category
        cat_dir = os.path.join(cache_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        videos = [f for f in os.listdir(vid_folder) if f.endswith(('.mp4', '.avi'))]
        print(f"\n👉 Đang xử lý thư mục {category} ({len(videos)} video):")
        
        for vid_name in tqdm(videos):
            vid_path = os.path.join(vid_folder, vid_name)
            xml_path = os.path.join(xml_folder, os.path.splitext(vid_name)[0] + '.xml')
            
            if not os.path.exists(xml_path):
                continue
            
            video_id = os.path.splitext(vid_name)[0]
            
            entries = extract_and_save_tracks(
                vid_path, xml_path, label,
                save_dir=cat_dir,
                video_id=video_id,
                img_size=img_size,
                target_fps=target_fps
            )
            manifest.extend(entries)
    
    if len(manifest) == 0:
        print("❌ Không trích xuất được track nào!")
        return manifest
    
    # Tính thống kê
    total_windows = 0
    for entry in manifest:
        n = entry["num_frames"]
        total_windows += max(0, (n - WINDOW_LENGTH) // WINDOW_STEP + 1)
    
    n0 = sum(1 for e in manifest if e["label"] == 0)
    n1 = sum(1 for e in manifest if e["label"] == 1)
    
    print(f"\n✅ Hoàn tất! {len(manifest)} tracks ({n0} non-violence, {n1} violence)")
    print(f"   📊 ~{total_windows} sliding windows (length={WINDOW_LENGTH}, step={WINDOW_STEP})")
    print(f"   → Tương thích Input RNN/LSTM: (samples, {WINDOW_LENGTH}, 3, {img_size}, {img_size})")
    
    # Lưu manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"💾 Đã lưu manifest vào '{manifest_path}'.")
    
    return manifest


def stratified_video_split(manifest, val_ratio=0.2, seed=42):
    """Chia manifest theo VIDEO (không theo track) với stratification.
    
    Đảm bảo:
    - Tất cả track từ cùng 1 video nằm cùng 1 set (train HOẶC val)
    - Tỷ lệ violence / non-violence cân bằng giữa train và val
    - Không có data leakage giữa train và val
    
    Returns:
        train_manifest, val_manifest
    """
    rng = np.random.RandomState(seed)
    
    # Nhóm tracks theo video nguồn
    # File path dạng: processed_tracks_v4/violence/vid_001_t0.npy
    video_groups = defaultdict(list)
    for entry in manifest:
        basename = os.path.basename(entry["file"])
        # Tách video_id: bỏ phần _tN.npy cuối
        video_id = basename.rsplit('_t', 1)[0]
        category = "violence" if entry["label"] == 1 else "non_violence"
        key = f"{category}/{video_id}"
        video_groups[key].append(entry)
    
    # Tách theo class
    nv_videos = {k: v for k, v in video_groups.items() if k.startswith("non_violence")}
    v_videos = {k: v for k, v in video_groups.items() if k.startswith("violence")}
    
    def split_class(group_dict, val_ratio):
        keys = list(group_dict.keys())
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * val_ratio))
        val_keys = keys[:n_val]
        train_keys = keys[n_val:]
        
        train_entries = []
        val_entries = []
        for k in train_keys:
            train_entries.extend(group_dict[k])
        for k in val_keys:
            val_entries.extend(group_dict[k])
        return train_entries, val_entries
    
    nv_train, nv_val = split_class(nv_videos, val_ratio)
    v_train, v_val = split_class(v_videos, val_ratio)
    
    train_manifest = nv_train + v_train
    val_manifest = nv_val + v_val
    
    print(f"\n📊 Stratified Video Split:")
    print(f"   Videos: NV={len(nv_videos)} ({len(nv_videos)-max(1,int(len(nv_videos)*val_ratio))} train, {max(1,int(len(nv_videos)*val_ratio))} val), "
          f"V={len(v_videos)} ({len(v_videos)-max(1,int(len(v_videos)*val_ratio))} train, {max(1,int(len(v_videos)*val_ratio))} val)")
    print(f"   Tracks: Train={len(train_manifest)}, Val={len(val_manifest)}")
    
    return train_manifest, val_manifest


# ==========================================
# 2. LAZY DATASET — SLIDING WINDOW ON-THE-FLY
#    (Không load hết vào RAM, chỉ đọc 30 frame/lần)
# ==========================================
class LazyWindowDataset(Dataset):
    """Dataset đọc dữ liệu từ disk + áp dụng sliding window on-the-fly.
    
    Thay vì load toàn bộ cửa sổ vào RAM (hàng chục GB), dataset này:
    - Lưu index: (track_idx, window_start) cho mỗi cửa sổ
    - Khi __getitem__: đọc file .npy bằng memory-map, lấy đúng 30 frame cần thiết
    - Mỗi lần chỉ tốn ~1.5MB RAM / sample
    
    Output shape: (30, 3, 64, 64) — tương thích Input RNN/LSTM.
    """
    def __init__(self, manifest, window_length=WINDOW_LENGTH, step=WINDOW_STEP, augment=True):
        self.manifest = manifest
        self.window_length = window_length
        self.augment = augment
        
        # Xây dựng index tất cả cửa sổ hợp lệ
        self.windows = []  # List of (manifest_idx, start_frame)
        for i, entry in enumerate(manifest):
            n = entry["num_frames"]
            if n < window_length:
                continue
            for start in range(0, n - window_length + 1, step):
                self.windows.append((i, start))
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        manifest_idx, start = self.windows[idx]
        entry = self.manifest[manifest_idx]
        
        # Đọc file .npy bằng memory-map (không load toàn bộ file vào RAM)
        track_data = np.load(entry["file"], mmap_mode='r')
        
        # Lấy đúng cửa sổ 30 frame
        window = track_data[start : start + self.window_length]
        x = torch.from_numpy(window.copy())  # copy() vì mmap_mode='r' là read-only
        # x shape: (30, 3, 64, 64)
        
        y = entry["label"]
        
        # Data Augmentation (chỉ khi training)
        if self.augment:
            # Random Horizontal Flip (cả sequence)
            if random.random() > 0.5:
                x = x.flip(-1)
            
            # Random Temporal Reverse (phát ngược sequence)
            # Hành vi bạo lực vẫn là bạo lực khi phát ngược → augment hữu ích
            if random.random() > 0.5:
                x = x.flip(0)
            
            # Random Speed Variation (tăng/giảm tốc độ)
            # Lấy ra subset frame rồi lặp/nội suy để giữ nguyên window_length
            if random.random() > 0.7:
                seq_len = x.size(0)
                speed = random.uniform(0.8, 1.2)
                new_len = int(seq_len * speed)
                if new_len >= seq_len:
                    # Slow-motion: lấy subsequence ngẫu nhiên seq_len frame từ new_len
                    indices = sorted(random.sample(range(new_len), seq_len))
                    indices = [min(i, seq_len - 1) for i in indices]
                    x = x[indices]
                else:
                    # Fast-forward: lặp lại frame để đủ seq_len
                    indices = [int(i * new_len / seq_len) for i in range(seq_len)]
                    indices = [min(i, seq_len - 1) for i in indices]
                    x = x[indices]
            
            # Random Brightness jitter
            if random.random() > 0.5:
                brightness = random.uniform(0.85, 1.15)
                x = torch.clamp(x * brightness, 0, 1)
            
            # Random Temporal Jitter: xóa ngẫu nhiên 1-2 frame rồi lặp frame kế bên
            if random.random() > 0.7:
                seq_len = x.size(0)
                num_drop = random.randint(1, 2)
                for _ in range(num_drop):
                    drop_idx = random.randint(1, seq_len - 2)
                    x[drop_idx] = x[drop_idx - 1]
            
            # Random Gaussian Noise nhẹ
            if random.random() > 0.5:
                noise = torch.randn_like(x) * 0.015
                x = torch.clamp(x + noise, 0, 1)
            
            # Random Erasing nhỏ (mô phỏng bị che khuất)
            if random.random() > 0.75:
                _, _, h, w = x.shape
                eh = random.randint(h // 8, h // 4)
                ew = random.randint(w // 8, w // 4)
                ey = random.randint(0, h - eh)
                ex = random.randint(0, w - ew)
                x[:, :, ey:ey+eh, ex:ex+ew] = 0
        
        return x, y


# ==========================================
# 3. MIXUP — CHỐNG OVERFITTING MẠNH
# ==========================================
def mixup_data(x, y, alpha=0.3):
    """MixUp: trộn ngẫu nhiên 2 mẫu trong batch để model học smoothly hơn.
    
    Thay vì học "ảnh A là violence", model học "70% ảnh A + 30% ảnh B".
    → Giảm overfitting đáng kể (thường +3-7% val acc).
    
    Args:
        x: input tensor (B, seq, C, H, W)
        y: labels tensor (B,)
        alpha: tham số Beta distribution (0.3 = trộn vừa phải)
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Đảm bảo lam >= 0.5 (mẫu gốc chiếm đa số)
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss cho MixUp: trung bình có trọng số giữa 2 label."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==========================================
# 4. KIẾN TRÚC MÔ HÌNH CẢI TIẾN
#    Pipeline: Spatial CNN → Multi-scale Conv1D → Bi-LSTM → Attention → FC
# ==========================================
class ImprovedViolenceModel(nn.Module):
    def __init__(self, seq_length=30, img_size=64, channels=3, num_classes=2):
        super(ImprovedViolenceModel, self).__init__()
        
        # === SPATIAL CNN - 4 blocks, BatchNorm, Deeper ===
        self.spatial_cnn = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
            
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),  # -> 2x2
        )
        
        cnn_out_dim = 256 * 2 * 2  # = 1024
        
        # Spatial Dropout — tắt cả feature map thay vì pixel đơn lẻ
        # Hiệu quả hơn Dropout thông thường cho CNN
        self.spatial_dropout = nn.Dropout2d(0.15)
        
        # FC giảm chiều + Dropout
        self.fc_reduce = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        
        # === MULTI-SCALE TEMPORAL Conv1D ===
        # Kernel 3: motion ngắn hạn (cú đấm, giật)
        self.temporal_conv_k3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        # Kernel 5: motion trung hạn (xô đẩy)
        self.temporal_conv_k5 = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        # Kernel 7: motion dài hạn (đuổi nhau, hành hung kéo dài)
        self.temporal_conv_k7 = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        # Concat: 128 + 64 + 64 = 256
        
        self.temporal_pool = nn.MaxPool1d(2)  # seq // 2
        
        # === BI-LSTM - 2 layers ===
        lstm_input = 256
        self.lstm_hidden = 128
        self.bilstm = nn.LSTM(
            lstm_input, self.lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # === TEMPORAL ATTENTION ===
        self.attention = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # === CLASSIFIER ===
        self.fc_classifier = nn.Sequential(
            nn.Linear(self.lstm_hidden * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        b, seq, c, h, w = x.size()
        
        # 1. Spatial CNN
        x = x.view(b * seq, c, h, w)
        x = self.spatial_cnn(x)
        x = self.spatial_dropout(x)  # Spatial Dropout sau CNN
        x = x.view(b * seq, -1)
        x = self.fc_reduce(x)
        x = x.view(b, seq, -1)  # (B, seq, 256)
        
        # 2. Multi-scale Temporal Conv1D
        x_t = x.permute(0, 2, 1)  # (B, 256, seq)
        t3 = self.temporal_conv_k3(x_t)
        t5 = self.temporal_conv_k5(x_t)
        t7 = self.temporal_conv_k7(x_t)
        x_t = torch.cat([t3, t5, t7], dim=1)  # (B, 256, seq)
        x_t = self.temporal_pool(x_t)          # (B, 256, seq//2)
        x_t = x_t.permute(0, 2, 1)            # (B, seq//2, 256)
        
        # 3. Bi-LSTM
        lstm_out, _ = self.bilstm(x_t)  # (B, seq//2, 256)
        
        # 4. Attention - tập trung vào timestep quan trọng nhất
        attn_weights = self.attention(lstm_out)        # (B, seq//2, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, 256)
        
        # 5. Classifier - trả ra raw logits (dùng cho CrossEntropyLoss)
        out = self.fc_classifier(context)
        return out
    
    def predict_proba(self, x):
        """Trả về xác suất (%) cho mỗi class sau khi qua Softmax.
        Output: tensor shape (B, 2) với cột 0 = % Non-Violence, cột 1 = % Violence
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs

# ==========================================
# 4. HÀM VẼ BIỂU ĐỒ
# ==========================================
def plot_history(train_losses, val_losses, train_accs, val_accs, filename='improved_history.png'):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Loss over Epochs', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy over Epochs (%)', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"📊 Đã lưu biểu đồ: {filename}")

def plot_cm(y_true, y_pred, classes, filename='improved_confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 16})
    plt.title('Confusion Matrix - Improved Model', fontsize=14)
    plt.ylabel('Thực tế', fontsize=12)
    plt.xlabel('Dự đoán', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"📊 Đã lưu confusion matrix: {filename}")

# ==========================================
# 5. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    # Seed cho reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Bắt đầu chạy trên: {device}")
    
    # === HYPERPARAMETERS ===
    IMG_SIZE = 64
    SEQ_LENGTH = 30
    BATCH_SIZE = 16
    EPOCHS = 80
    LR = 3e-4
    WEIGHT_DECAY = 5e-4      # L2 regularization (tăng để chống overfitting)
    PATIENCE = 20             # Early stopping (tăng vì MixUp hội tụ chậm hơn)
    VIOLENCE_THRESHOLD = 0.45 # Ngưỡng 45%
    
    # ==============================================================
    # 1. TIỀN XỬ LÝ — Lưu track crops ra disk (chỉ chạy 1 lần)
    # ==============================================================
    DATA_DIR = "data"
    LABELS_DIR = "fix_labels"
    CACHE_DIR = "processed_tracks_v4"
    
    manifest = prepare_dataset(DATA_DIR, LABELS_DIR, cache_dir=CACHE_DIR, img_size=IMG_SIZE)
    
    if len(manifest) == 0:
        print("❌ Lỗi: Không trích xuất được track nào.")
        exit()
    
    # ==============================================================
    # 2. TẠO DATASET VỚI SLIDING WINDOW ON-THE-FLY
    # ==============================================================
    # Chia theo VIDEO (không theo track) với stratification
    # → Đảm bảo tracks cùng video không bị leak giữa train/val
    # → Đảm bảo tỷ lệ violence/non-violence cân bằng
    train_manifest, val_manifest = stratified_video_split(manifest, val_ratio=0.2, seed=SEED)
    
    train_dataset = LazyWindowDataset(train_manifest, augment=True)
    val_dataset = LazyWindowDataset(val_manifest, augment=False)
    
    # Thống kê
    n_train_nv = sum(1 for e in train_manifest if e["label"] == 0)
    n_train_v = sum(1 for e in train_manifest if e["label"] == 1)
    n_val_nv = sum(1 for e in val_manifest if e["label"] == 0)
    n_val_v = sum(1 for e in val_manifest if e["label"] == 1)
    
    print(f"   Train: {len(train_dataset)} windows ({n_train_nv} NV tracks, {n_train_v} V tracks)")
    print(f"   Val:   {len(val_dataset)} windows ({n_val_nv} NV tracks, {n_val_v} V tracks)")
    
    # Class Weights (tính từ tổng windows mỗi class)
    train_labels = [train_manifest[mi]["label"] for mi, _ in train_dataset.windows]
    n_nv_win = train_labels.count(0)
    n_v_win = train_labels.count(1)
    total_win = len(train_labels)
    
    if n_nv_win > 0 and n_v_win > 0:
        class_weights = torch.tensor(
            [total_win / (2 * n_nv_win), total_win / (2 * n_v_win)],
            dtype=torch.float32
        ).to(device)
    else:
        class_weights = torch.ones(2, dtype=torch.float32).to(device)
    
    print(f"⚖️  Class weights: {class_weights.cpu().numpy()}")
    print(f"   (NV windows: {n_nv_win}, V windows: {n_v_win})")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    print(f"📦 Train: {len(train_dataset)} mẫu, Val: {len(val_dataset)} mẫu")
    
    # # ==============================================================
    # # 3. KHỞI TẠO MODEL
    # # ==============================================================
    # model = ImprovedViolenceModel(seq_length=SEQ_LENGTH, img_size=IMG_SIZE).to(device)
    
    # # Đếm parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"🧠 Model: {total_params:,} params ({trainable_params:,} trainable)")
    
    # # Label Smoothing CrossEntropy + Class Weights
    # # Label smoothing 0.2: soft target [0.1, 0.9] thay vì [0, 1] → giảm overconfidence
    # criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
    
    # # AdamW optimizer
    # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # # Cosine Annealing LR Scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    
    # # ==============================================================
    # # 4. TRAINING LOOP
    # # ==============================================================
    # train_losses, val_losses, train_accs, val_accs = [], [], [], []
    # best_val_acc = 0.0
    # best_model_state = None
    # patience_counter = 0
    
    # print(f"\n🚀 BẮT ĐẦU HUẤN LUYỆN - {EPOCHS} epochs, Early Stopping patience={PATIENCE}")
    # print("=" * 100)
    
    # for epoch in range(EPOCHS):
    #     # --- TRAIN ---
    #     model.train()
    #     running_loss, correct, total = 0.0, 0, 0
        
    #     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    #     for inputs, labels in pbar:
    #         inputs = inputs.to(device, dtype=torch.float32)
    #         labels = labels.to(device, dtype=torch.long)
            
    #         # === MIXUP: trộn 2 mẫu ngẫu nhiên trong batch ===
    #         inputs_mixed, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.3)
            
    #         optimizer.zero_grad()
    #         outputs = model(inputs_mixed)
    #         loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
    #         loss.backward()
            
    #         # Gradient Clipping
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
    #         optimizer.step()
            
    #         running_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         # Accuracy tính dựa trên label chính (lam >= 0.5 nên labels_a chiếm đa số)
    #         total += labels.size(0)
    #         correct += (lam * (predicted == labels_a).sum().item()
    #                    + (1 - lam) * (predicted == labels_b).sum().item())
            
    #         pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")
            
    #     train_loss = running_loss / len(train_loader)
    #     train_acc = 100 * correct / total
        
    #     # --- VALIDATION ---
    #     model.eval()
    #     val_loss_sum, correct, total = 0.0, 0, 0
    #     all_preds, all_labels_list = [], []
        
    #     with torch.no_grad():
    #         for inputs, labels in val_loader:
    #             inputs = inputs.to(device, dtype=torch.float32)
    #             labels = labels.to(device, dtype=torch.long)
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)
                
    #             val_loss_sum += loss.item()
    #             # Dùng softmax + ngưỡng 45% thay vì argmax
    #             probs = torch.softmax(outputs, dim=1)
    #             predicted = (probs[:, 1] >= VIOLENCE_THRESHOLD).long()
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #             all_preds.extend(predicted.cpu().numpy())
    #             all_labels_list.extend(labels.cpu().numpy())
                
    #     val_loss = val_loss_sum / len(val_loader)
    #     val_acc = 100 * correct / total
        
    #     # Update LR
    #     scheduler.step()
    #     current_lr = optimizer.param_groups[0]['lr']
        
    #     train_losses.append(train_loss)
    #     val_losses.append(val_loss)
    #     train_accs.append(train_acc)
    #     val_accs.append(val_acc)
        
    #     # Tracking best model
    #     marker = ""
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         best_model_state = copy.deepcopy(model.state_dict())
    #         patience_counter = 0
    #         marker = " ⭐ BEST"
    #         torch.save(model.state_dict(), "best_model_v2.pth")
    #     else:
    #         patience_counter += 1
        
    #     print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
    #           f"Loss: {train_loss:.4f} | Acc: {train_acc:.1f}% || "
    #           f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
    #           f"LR: {current_lr:.2e} | P: {patience_counter}/{PATIENCE}{marker}")
        
    #     # Early Stopping
    #     if patience_counter >= PATIENCE:
    #         print(f"\n⏹️  Early Stopping tại epoch {epoch+1}! Best Val Acc: {best_val_acc:.2f}%")
    #         break
    
    # # ==============================================================
    # # 5. LOAD BEST MODEL VÀ ĐÁNH GIÁ CUỐI CÙNG
    # # ==============================================================
    # print("\n" + "=" * 100)
    # print(f"🏆 Load lại model tốt nhất (Val Acc = {best_val_acc:.2f}%) để đánh giá cuối cùng...")
    # model.load_state_dict(best_model_state)
    # model.eval()
    
    # all_preds, all_labels_list, all_probs = [], [], []
    # with torch.no_grad():
    #     for inputs, labels in val_loader:
    #         inputs = inputs.to(device, dtype=torch.float32)
    #         labels = labels.to(device, dtype=torch.long)
    #         probs = model.predict_proba(inputs)  # Dùng softmax
    #         predicted = (probs[:, 1] >= VIOLENCE_THRESHOLD).long()  # Ngưỡng 45%
    #         all_preds.extend(predicted.cpu().numpy())
    #         all_labels_list.extend(labels.cpu().numpy())
    #         all_probs.extend(probs[:, 1].cpu().numpy())  # Lưu xác suất violence
    
    # final_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels_list))
    
    # # In một vài ví dụ minh họa
    # print(f"\n🔍 VÍ DỤ OUTPUT (ngưỡng = {VIOLENCE_THRESHOLD*100:.0f}%):")
    # for i in range(min(10, len(all_probs))):
    #     violence_pct = all_probs[i] * 100
    #     actual = 'Violence' if all_labels_list[i] == 1 else 'Non-Violence'
    #     pred = 'Violence' if all_preds[i] == 1 else 'Non-Violence'
    #     status = '✅' if all_preds[i] == all_labels_list[i] else '❌'
    #     print(f"  {status} Xác suất BL: {violence_pct:.1f}% → Dự đoán: {pred} | Thực tế: {actual}")
    
    # # 6. XUẤT KẾT QUẢ
    # plot_history(train_losses, val_losses, train_accs, val_accs)
    # plot_cm(all_labels_list, all_preds, classes=['Non-Violence', 'Violence'])
    
    # print("\n📋 BÁO CÁO PHÂN LOẠI (Best Model):")
    # print(classification_report(all_labels_list, all_preds, target_names=['Non-Violence', 'Violence']))
    
    # print(f"\n⚙️  Ngưỡng phân loại Violence: {VIOLENCE_THRESHOLD*100:.0f}%")
    # print(f"💾 Model tốt nhất đã được lưu tại 'best_model_v2.pth'")
    # print(f"🎯 Final Validation Accuracy: {final_acc:.2f}%")
    # print("✅ HOÀN TẤT!")