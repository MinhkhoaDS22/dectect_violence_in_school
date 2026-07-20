"""
evaluate_model.py — Đánh giá model violence detection trên tập test bên ngoài

3 hướng tiếp cận:
  Hướng 1: Sliding Window   (window-level, IMG_SIZE=96 — giống training)
  Hướng 2: Full App Pipeline (video-level,  IMG_SIZE=64 — giống backend)
  Hướng 3: No Sliding Window (track-level,  IMG_SIZE=96 — giống training, 30 frame đầu)

Input : data_test/violence/  &  data_test/non_violence/
Output: biểu đồ, ma trận nhầm lẫn, report text  →  test_result/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — tránh lỗi trên server
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from ultralytics import YOLO
from train_ai import ImprovedViolenceModel
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from tqdm import tqdm
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════
# HẰNG SỐ
# ═══════════════════════════════════════════════════════════════════════
TARGET_FPS        = 30
WINDOW_LENGTH     = 30        # 30 frame / cửa sổ (= 1 giây ở 30 fps)
WINDOW_STEP       = 5         # Bước nhảy sliding window
VIOLENCE_THRESHOLD = 0.45     # Ngưỡng 45 %
CONSECUTIVE_WINDOWS = 5       # Hướng 2: ≥ 5 slot liên tiếp vượt ngưỡng
APPROACH1_VIDEO_THRESHOLD = 0.45   # Hướng 1: > 50% windows violence → video violence
YOLO_IMGSZ        = 320
YOLO_FRAME_SKIP   = 2         # Chạy YOLO mỗi 2 frame
BATCH_INFERENCE   = 8

APPROACH1_IMG_SIZE = 96       # Giống training (train_ai.py)
APPROACH2_IMG_SIZE = 64       # Giống backend  (violence_detector.py)
APPROACH3_IMG_SIZE = 96       # Giống training

MODEL_PATH = os.path.join("results", "best_model.pth")
YOLO_PATH  = "yolo11s.pt"
TEST_DIR   = "Test"
OUTPUT_DIR = "test_result"

CLASS_NAMES = ["Non-Violence", "Violence"]


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 1 — HÀM TIỆN ÍCH CHUNG
# ═══════════════════════════════════════════════════════════════════════

def collect_test_videos(test_dir):
    """Quét data_test/ → list (path, label, display_name).

    Chỉ yêu cầu mỗi folder (violence, non_violence) có ≥ 1 video.
    """
    videos = []
    for category, label in [("violence", 1), ("non_violence", 0)]:
        folder = os.path.join(test_dir, category)
        if not os.path.exists(folder):
            print(f"⚠️  Folder '{folder}' không tồn tại, bỏ qua.")
            continue
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".mp4", ".avi"))
        )
        for f in files:
            videos.append((os.path.join(folder, f), label, f"{category}/{f}"))

    v_count  = sum(1 for _, l, _ in videos if l == 1)
    nv_count = sum(1 for _, l, _ in videos if l == 0)

    if v_count < 1:
        raise RuntimeError("Folder violence phải có ít nhất 1 video!")
    if nv_count < 1:
        raise RuntimeError("Folder non_violence phải có ít nhất 1 video!")

    print(f"📁 Dataset: {len(videos)} video "
          f"({v_count} violence, {nv_count} non-violence)")
    return videos


def normalize_frame_indices(original_fps, target_fps, total_frames):
    """Chuẩn hoá FPS — giống train_ai.py & violence_detector.py."""
    if original_fps <= 0 or total_frames <= 0:
        return list(range(total_frames))
    duration = total_frames / original_fps
    target_total = int(round(duration * target_fps))
    if target_total <= 0:
        return list(range(total_frames))
    indices = []
    for i in range(target_total):
        orig_idx = int(round(i * (total_frames - 1) / max(target_total - 1, 1)))
        indices.append(min(orig_idx, total_frames - 1))
    return indices


def process_video_yolo(video_path, yolo_model, img_sizes, target_fps=TARGET_FPS):
    """Chạy YOLO + ByteTrack **1 lần** trên video, crop ở nhiều kích thước.

    Args:
        video_path : đường dẫn video
        yolo_model : model YOLO đã load
        img_sizes  : list kích thước crop, vd [96, 64]

    Returns:
        results        : {img_size: {track_crops: {tid: [chw]},
                                      track_times: {tid: [sec]}}}
        video_duration : float (giây)
    """
    cap = cv2.VideoCapture(video_path)
    empty = {sz: {"track_crops": {}, "track_times": {}} for sz in img_sizes}
    if not cap.isOpened():
        return empty, 0.0

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / original_fps if original_fps > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        return empty, 0.0

    # Chuẩn hoá FPS
    if abs(original_fps - target_fps) > 0.5:
        resampled = normalize_frame_indices(original_fps, target_fps, total_frames)
    else:
        resampled = list(range(total_frames))

    resampled_set = set(resampled)
    orig_to_time  = {oi: i / target_fps for i, oi in enumerate(resampled)}

    tracks_boxes = {}                       # {tid: {frame_idx: (x1,y1,x2,y2)}}
    results = {
        sz: {"track_crops": {}, "track_times": {}}
        for sz in img_sizes
    }

    is_first_track = True
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLO detect + ByteTrack ──────────────────────────────
        if frame_idx % YOLO_FRAME_SKIP == 0:
            try:
                det = yolo_model.track(
                    frame,
                    persist=not is_first_track,   # False lần đầu → reset tracker
                    conf=0.38, iou=0.55,
                    classes=[0],
                    tracker="bytetrack.yaml",
                    verbose=False, imgsz=YOLO_IMGSZ,
                )
                is_first_track = False

                if (det[0].boxes is not None
                        and det[0].boxes.id is not None):
                    for box, tid in zip(
                        det[0].boxes.xyxy.cpu().numpy(),
                        det[0].boxes.id.int().cpu().numpy(),
                    ):
                        tid = int(tid)
                        tracks_boxes.setdefault(tid, {})[frame_idx] = \
                            tuple(box.tolist())
            except Exception:
                pass                             # Bỏ qua frame lỗi

        # ── Crop ở các kích thước ────────────────────────────────
        if frame_idx in resampled_set:
            t_sec = orig_to_time.get(frame_idx, frame_idx / original_fps)
            h_img, w_img = frame.shape[:2]

            for tid, fdict in tracks_boxes.items():
                if frame_idx not in fdict:
                    continue
                x1, y1, x2, y2 = fdict[frame_idx]
                bw, bh = x2 - x1, y2 - y1
                cx1 = max(0,     int(x1 - 0.05 * bw))
                cy1 = max(0,     int(y1 - 0.05 * bh))
                cx2 = min(w_img, int(x2 + 0.05 * bw))
                cy2 = min(h_img, int(y2 + 0.05 * bh))

                if cx2 > cx1 and cy2 > cy1:
                    roi = frame[cy1:cy2, cx1:cx2]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                    for sz in img_sizes:
                        crop = cv2.resize(roi_rgb, (sz, sz),
                                          interpolation=cv2.INTER_LINEAR)
                        crop_chw = np.transpose(
                            crop.astype(np.float32) / 255.0, (2, 0, 1)
                        )
                        results[sz]["track_crops"] \
                            .setdefault(tid, []).append(crop_chw)
                        results[sz]["track_times"] \
                            .setdefault(tid, []).append(t_sec)

        frame_idx += 1

    cap.release()
    return results, video_duration


def batch_inference(model, windows, device):
    """Chạy inference theo batch, trả về list[float] probs."""
    probs_out = []
    model.eval()
    with torch.no_grad():
        for bi in range(0, len(windows), BATCH_INFERENCE):
            batch = windows[bi: bi + BATCH_INFERENCE]
            x = torch.from_numpy(np.stack(batch)).to(device, dtype=torch.float32)
            probs = model.predict_proba(x)[:, 1].cpu().numpy()
            probs_out.extend(probs.tolist())
    return probs_out


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 2 — ĐÁNH GIÁ TỪNG VIDEO QUA 3 HƯỚNG
# ═══════════════════════════════════════════════════════════════════════

def evaluate_approach1(track_crops, label, model, device):
    """Hướng 1 — Sliding window → window-level probs + video-level pred.

    Video-level: nếu > 50% windows dự đoán violence → video = violence.

    Returns:
        all_probs  : list[float] — prob mỗi window
        all_labels : list[int]   — label mỗi window (= label video)
        video_pred : int         — 0 hoặc 1 (kết luận video-level)
    """
    valid = {tid: c for tid, c in track_crops.items()
             if len(c) >= WINDOW_LENGTH}

    all_probs, all_labels = [], []
    for tid, crops in valid.items():
        arr = np.stack(crops, axis=0)
        n = arr.shape[0]
        windows = [arr[s: s + WINDOW_LENGTH]
                    for s in range(0, n - WINDOW_LENGTH + 1, WINDOW_STEP)]
        if not windows:
            continue
        probs = batch_inference(model, windows, device)
        all_probs.extend(probs)
        all_labels.extend([label] * len(probs))

    # Video-level prediction: > 50% windows violence → video violence
    if all_probs:
        n_violence = sum(1 for p in all_probs if p >= VIOLENCE_THRESHOLD)
        violence_ratio = n_violence / len(all_probs)
        video_pred = 1 if violence_ratio > APPROACH1_VIDEO_THRESHOLD else 0
    else:
        violence_ratio = 0.0
        video_pred = 0   # default non-violence khi không có data

    return all_probs, all_labels, video_pred, violence_ratio


def evaluate_approach2(track_crops, track_times, label, model, device):
    """Hướng 2 — Full pipeline (giống violence_detector.py).

    Returns dict chứa:
        pred, label, is_violence,
        window_probs, window_labels,
        violence_ratio, max_violent_persons, segments, note
    """
    valid = {tid: c for tid, c in track_crops.items()
             if len(c) >= WINDOW_LENGTH}

    no_data = {
        "pred": 0, "label": label, "is_violence": False,
        "window_probs": [], "window_labels": [],
        "violence_ratio": 0.0, "max_violent_persons": 0, "segments": 0,
    }
    if not valid:
        no_data["note"] = "Không có track đủ dài"
        return no_data

    # ── Inference từng track ──────────────────────────────────
    all_window_results = []     # [(center_time, prob, t_s, t_e)]
    window_probs, window_labels = [], []

    for tid, crops in valid.items():
        arr   = np.stack(crops, axis=0)
        times = track_times.get(tid, [])
        n     = arr.shape[0]

        windows, starts = [], []
        for s in range(0, n - WINDOW_LENGTH + 1, WINDOW_STEP):
            windows.append(arr[s: s + WINDOW_LENGTH])
            starts.append(s)

        if not windows:
            continue

        probs = batch_inference(model, windows, device)
        window_probs.extend(probs)
        window_labels.extend([label] * len(probs))

        for s, prob in zip(starts, probs):
            t_s = times[s] if s < len(times) else 0
            t_e = times[min(s + WINDOW_LENGTH - 1, len(times) - 1)] \
                  if times else 0
            t_center = (t_s + t_e) / 2.0
            all_window_results.append((t_center, prob, t_s, t_e))

    if not all_window_results:
        no_data["note"] = "Không trích xuất được window"
        return no_data

    # ── Time-slot MAX aggregation (giống violence_detector.py) ─
    slot_duration = WINDOW_STEP / TARGET_FPS
    slot_data = defaultdict(list)
    for t_center, prob, t_s, t_e in all_window_results:
        slot_data[int(t_center / slot_duration)].append((prob, t_s, t_e))

    max_slot = max(slot_data.keys())
    timeline = []
    for s in range(max_slot + 1):
        if s in slot_data:
            entries = slot_data[s]
            max_score = float(np.max([p for p, _, _ in entries]))
            n_violent = sum(1 for p, _, _ in entries if p >= VIOLENCE_THRESHOLD)
            timeline.append((s, max_score, n_violent, len(entries)))
        else:
            timeline.append((s, 0.0, 0, 0))

    # ── Consecutive-windows check ─────────────────────────────
    is_violence = False
    violent_segments_count = 0
    run_count = 0

    for _, score, _, _ in timeline:
        if score >= VIOLENCE_THRESHOLD:
            run_count += 1
        else:
            if run_count >= CONSECUTIVE_WINDOWS:
                is_violence = True
                violent_segments_count += 1
            run_count = 0

    if run_count >= CONSECUTIVE_WINDOWS:
        is_violence = True
        violent_segments_count += 1

    n_violent_slots = sum(1 for _, sc, _, _ in timeline
                         if sc >= VIOLENCE_THRESHOLD)
    violence_ratio = n_violent_slots / max(len(timeline), 1)
    max_violent_persons = max((nv for _, _, nv, _ in timeline), default=0)

    return {
        "pred": 1 if is_violence else 0,
        "label": label,
        "is_violence": is_violence,
        "window_probs": window_probs,
        "window_labels": window_labels,
        "violence_ratio": round(violence_ratio, 3),
        "max_violent_persons": max_violent_persons,
        "segments": violent_segments_count,
    }


def evaluate_approach3(track_crops, label, model, device):
    """Hướng 3 — Không sliding window, lấy 30 frame đầu mỗi track.

    Returns (probs, labels) — mỗi phần tử ứng với 1 track.
    """
    valid = {tid: c for tid, c in track_crops.items()
             if len(c) >= WINDOW_LENGTH}

    if not valid:
        return [], []

    samples = [np.stack(crops[:WINDOW_LENGTH], axis=0)
               for crops in valid.values()]

    probs  = batch_inference(model, samples, device)
    labels = [label] * len(probs)
    return probs, labels


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 3 — VẼ BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, title, filepath):
    """Ma trận nhầm lẫn dạng heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 18})
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Thực tế", fontsize=12)
    plt.xlabel("Dự đoán", fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  📊 Đã lưu: {os.path.basename(filepath)}")


def plot_prob_distribution(probs, labels, title, filepath):
    """Histogram phân phối xác suất violence."""
    probs_nv = [p for p, l in zip(probs, labels) if l == 0]
    probs_v  = [p for p, l in zip(probs, labels) if l == 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    if probs_nv:
        ax.hist(probs_nv, bins=40, alpha=0.6, color="steelblue",
                label="Non-Violence", density=True)
    if probs_v:
        ax.hist(probs_v, bins=40, alpha=0.6, color="tomato",
                label="Violence", density=True)
    ax.axvline(VIOLENCE_THRESHOLD, color="black", linestyle="--",
               linewidth=2, label=f"Ngưỡng = {VIOLENCE_THRESHOLD*100:.0f}%")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("P(Violence)"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  📊 Đã lưu: {os.path.basename(filepath)}")


def plot_per_video_bar(per_video, title, filepath, metric_key="accuracy"):
    """Bar chart accuracy / metric từng video (Hướng 1 & 3)."""
    from matplotlib.patches import Patch

    names  = [d["name"] for d in per_video]
    labels = [d["label"] for d in per_video]
    values = [d.get(metric_key, 0) for d in per_video]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.6), 6))
    colors = ["tomato" if l == 1 else "steelblue" for l in labels]
    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor="tomato",    alpha=0.85, label="Violence"),
        Patch(facecolor="steelblue", alpha=0.85, label="Non-Violence"),
    ]
    ax.legend(handles=legend_elements)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  📊 Đã lưu: {os.path.basename(filepath)}")


def plot_per_video_approach1(per_video, filepath):
    """Bar chart Hướng 1 — violence_window_ratio, màu theo đúng/sai."""
    from matplotlib.patches import Patch

    names   = [d["name"] for d in per_video]
    ratios  = [d["violence_window_ratio"] * 100 for d in per_video]
    correct = [d["pred"] == d["label"] for d in per_video]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.6), 6))
    colors = ["#4CAF50" if c else "#F44336" for c in correct]
    bars = ax.bar(range(len(names)), ratios, color=colors, alpha=0.85)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Violence Windows (%)")
    ax.set_title("% Windows Violence từng Video — Hướng 1: Sliding Window",
                 fontsize=13, fontweight="bold")
    ax.axhline(APPROACH1_VIDEO_THRESHOLD * 100, color="black", linestyle="--",
               linewidth=1.5, label=f"Ngưỡng = {APPROACH1_VIDEO_THRESHOLD*100:.0f}%")
    ax.grid(axis="y", alpha=0.3)

    for bar, ratio, c in zip(bars, ratios, correct):
        icon = "✅" if c else "❌"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{icon} {ratio:.1f}%", ha="center", va="bottom", fontsize=7)

    legend_elements = [
        Patch(facecolor="#4CAF50", alpha=0.85, label="Dự đoán đúng"),
        Patch(facecolor="#F44336", alpha=0.85, label="Dự đoán sai"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  📊 Đã lưu: {os.path.basename(filepath)}")


def plot_per_video_approach2(per_video, filepath):
    """Bar chart Hướng 2 — violence_ratio, màu theo đúng/sai."""
    from matplotlib.patches import Patch

    names  = [d["name"]  for d in per_video]
    ratios = [d["violence_ratio"] * 100 for d in per_video]
    correct = [d["pred"] == d["label"] for d in per_video]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.6), 6))
    colors = ["#4CAF50" if c else "#F44336" for c in correct]
    bars = ax.bar(range(len(names)), ratios, color=colors, alpha=0.85)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Violence Ratio (%)")
    ax.set_title("Violence Ratio từng Video — Hướng 2: Full Pipeline",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, ratio, c in zip(bars, ratios, correct):
        icon = "✅" if c else "❌"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{icon} {ratio:.1f}%", ha="center", va="bottom", fontsize=7)

    legend_elements = [
        Patch(facecolor="#4CAF50", alpha=0.85, label="Dự đoán đúng"),
        Patch(facecolor="#F44336", alpha=0.85, label="Dự đoán sai"),
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  📊 Đã lưu: {os.path.basename(filepath)}")


def plot_comparison(metrics_dict, filepath):
    """So sánh 3 hướng side-by-side."""
    approaches = list(metrics_dict.keys())
    metric_names = ["Accuracy", "F1 (Violence)", "Precision (V)", "Recall (V)"]
    metric_keys  = ["accuracy", "f1_violence", "precision_violence", "recall_violence"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_names))
    width = 0.25
    palette = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, approach in enumerate(approaches):
        m = metrics_dict[approach]
        values = [m.get(k, 0) for k in metric_keys]
        bars = ax.bar(x + i * width, values, width,
                      label=approach, color=palette[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Phần trăm (%)")
    ax.set_title("So sánh 3 Hướng Đánh Giá", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 115)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  📊 Đã lưu: {os.path.basename(filepath)}")


def compute_metrics(y_true, y_pred):
    """Tính Accuracy, F1, Precision, Recall từ y_true / y_pred."""
    if not y_true:
        return {"accuracy": 0, "f1_violence": 0, "precision_violence": 0,
                "recall_violence": 0, "f1_non_violence": 0}

    acc = accuracy_score(y_true, y_pred) * 100
    rpt = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    return {
        "accuracy":            round(acc, 2),
        "f1_violence":         round(rpt["Violence"]["f1-score"] * 100, 2),
        "precision_violence":  round(rpt["Violence"]["precision"] * 100, 2),
        "recall_violence":     round(rpt["Violence"]["recall"] * 100, 2),
        "f1_non_violence":     round(rpt["Non-Violence"]["f1-score"] * 100, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 4 — XUẤT REPORT TEXT
# ═══════════════════════════════════════════════════════════════════════

def write_report(
    output_path, videos,
    a1_probs, a1_labels, a1_per_video,
    a2_details,
    a2_all_wprobs, a2_all_wlabels,
    a3_probs, a3_labels, a3_per_video,
    metrics1, metrics2, metrics3,
    device_name,
):
    """Ghi toàn bộ kết quả ra file text."""

    a1_preds = [1 if p >= VIOLENCE_THRESHOLD else 0 for p in a1_probs]
    a3_preds = [1 if p >= VIOLENCE_THRESHOLD else 0 for p in a3_probs]
    a2_vpreds  = [r["pred"]  for r in a2_details]
    a2_vlabels = [r["label"] for r in a2_details]

    v_cnt  = sum(1 for _, l, _ in videos if l == 1)
    nv_cnt = sum(1 for _, l, _ in videos if l == 0)

    with open(output_path, "w", encoding="utf-8") as f:
        # ── Header ───────────────────────────────────────────────
        f.write("═" * 70 + "\n")
        f.write("       ĐÁNH GIÁ MODEL — BÁO CÁO CHI TIẾT\n")
        f.write("═" * 70 + "\n\n")
        f.write(f"Ngày      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model     : {MODEL_PATH}\n")
        f.write(f"YOLO      : {YOLO_PATH}\n")
        f.write(f"Device    : {device_name}\n")
        f.write(f"Ngưỡng    : {VIOLENCE_THRESHOLD*100:.0f}%\n")
        f.write(f"Dataset   : {len(videos)} video "
                f"({v_cnt} violence, {nv_cnt} non-violence)\n")
        f.write(f"Thư mục   : {TEST_DIR}/\n\n")

        # ── HƯỚNG 1 ─────────────────────────────────────────────
        a1_vpreds  = [d["pred"]  for d in a1_per_video]
        a1_vlabels = [d["label"] for d in a1_per_video]

        f.write("─" * 70 + "\n")
        f.write("HƯỚNG 1: SLIDING WINDOW "
                f"(Video-level, IMG_SIZE={APPROACH1_IMG_SIZE})\n")
        f.write("─" * 70 + "\n\n")
        f.write(f"Tiêu chí    : > {APPROACH1_VIDEO_THRESHOLD*100:.0f}% "
                f"windows violence → video = violence\n")
        f.write(f"Tổng video  : {len(a1_per_video)}\n")
        f.write(f"Tổng windows: {len(a1_probs)}\n")
        f.write(f"Accuracy    : {metrics1['accuracy']:.2f}%\n\n")

        if a1_per_video:
            f.write("Classification Report:\n")
            f.write(classification_report(
                a1_vlabels, a1_vpreds,
                target_names=CLASS_NAMES, zero_division=0))
            f.write("\n")

        f.write("Chi tiết từng video:\n")
        for d in a1_per_video:
            lbl  = "Violence" if d["label"] == 1 else "Non-Violence"
            pred = "Violence" if d["pred"]  == 1 else "Non-Violence"
            icon = "✅" if d["pred"] == d["label"] else "❌"
            note = d.get("note", "")
            if note:
                f.write(f"  {icon} {d['name']} [{lbl}] → Pred: {pred} "
                        f"| {note}\n")
            else:
                vr = d['violence_window_ratio']
                f.write(f"  {icon} {d['name']} [{lbl}] → Pred: {pred} "
                        f"| {d['violence_windows']}/{d['total_windows']} "
                        f"windows violence ({vr*100:.1f}%)\n")

        # ── HƯỚNG 2 ─────────────────────────────────────────────
        f.write("\n" + "─" * 70 + "\n")
        f.write("HƯỚNG 2: FULL APP PIPELINE "
                f"(Video-level, IMG_SIZE={APPROACH2_IMG_SIZE})\n")
        f.write("─" * 70 + "\n\n")
        f.write(f"Tổng video   : {len(a2_details)}\n")
        f.write(f"Accuracy     : {metrics2['accuracy']:.2f}%\n\n")

        if a2_details:
            f.write("Classification Report:\n")
            f.write(classification_report(
                a2_vlabels, a2_vpreds,
                target_names=CLASS_NAMES, zero_division=0))
            f.write("\n")

        f.write("Chi tiết từng video:\n")
        for r in a2_details:
            lbl  = "Violence" if r["label"] == 1 else "Non-Violence"
            pred = "Violence" if r["pred"]  == 1 else "Non-Violence"
            icon = "✅" if r["pred"] == r["label"] else "❌"
            note = r.get("note", "")
            f.write(f"  {icon} {r['name']} [{lbl}] → Pred: {pred}")
            if note:
                f.write(f" | {note}\n")
            else:
                f.write(f" | ratio={r['violence_ratio']:.3f}"
                        f" | max_persons={r['max_violent_persons']}"
                        f" | segments={r['segments']}\n")

        # ── HƯỚNG 3 ─────────────────────────────────────────────
        f.write("\n" + "─" * 70 + "\n")
        f.write("HƯỚNG 3: NO SLIDING WINDOW "
                f"(Track-level, IMG_SIZE={APPROACH3_IMG_SIZE})\n")
        f.write("─" * 70 + "\n\n")
        f.write(f"Tổng tracks  : {len(a3_probs)}\n")
        f.write(f"Accuracy     : {metrics3['accuracy']:.2f}%\n\n")

        if a3_probs:
            f.write("Classification Report:\n")
            f.write(classification_report(
                a3_labels, a3_preds,
                target_names=CLASS_NAMES, zero_division=0))
            f.write("\n")

        f.write("Chi tiết từng video:\n")
        for d in a3_per_video:
            lbl = "Violence" if d["label"] == 1 else "Non-Violence"
            note = d.get("note", "")
            if note:
                f.write(f"  {d['name']} [{lbl}]: {note}\n")
            else:
                f.write(f"  {d['name']} [{lbl}]: "
                        f"{d['total_tracks']} tracks, "
                        f"{d['correct']} đúng ({d['accuracy']:.1f}%)\n")

        # ── SO SÁNH ─────────────────────────────────────────────
        f.write("\n" + "─" * 70 + "\n")
        f.write("SO SÁNH 3 HƯỚNG\n")
        f.write("─" * 70 + "\n\n")

        hdr = f"{'Metric':<25} {'Hướng 1':>10} {'Hướng 2':>10} {'Hướng 3':>10}\n"
        f.write(hdr)
        f.write("-" * 55 + "\n")

        rows = [
            ("Accuracy (%)",
             metrics1["accuracy"], metrics2["accuracy"], metrics3["accuracy"]),
            ("F1 Violence (%)",
             metrics1["f1_violence"], metrics2["f1_violence"],
             metrics3["f1_violence"]),
            ("Precision V (%)",
             metrics1["precision_violence"], metrics2["precision_violence"],
             metrics3["precision_violence"]),
            ("Recall V (%)",
             metrics1["recall_violence"], metrics2["recall_violence"],
             metrics3["recall_violence"]),
            ("F1 Non-Violence (%)",
             metrics1["f1_non_violence"], metrics2["f1_non_violence"],
             metrics3["f1_non_violence"]),
        ]
        for name, v1, v2, v3 in rows:
            f.write(f"{name:<25} {v1:>10.2f} {v2:>10.2f} {v3:>10.2f}\n")

        f.write("\n" + "═" * 70 + "\n")
        f.write("KẾT THÚC BÁO CÁO\n")
        f.write("═" * 70 + "\n")

    print(f"  📄 Đã lưu: {os.path.basename(output_path)}")


# ═══════════════════════════════════════════════════════════════════════
# PHẦN 5 — MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 70)
    print("🔬 EVALUATE MODEL — 3 HƯỚNG TIẾP CẬN")
    print("═" * 70)

    # ── Setup ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Thu thập video test ──────────────────────────────────────
    videos = collect_test_videos(TEST_DIR)

    # ── Load model ───────────────────────────────────────────────
    print(f"\n📦 Loading model: {MODEL_PATH}")
    model = ImprovedViolenceModel(seq_length=WINDOW_LENGTH,
                                  img_size=APPROACH1_IMG_SIZE)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("✅ Model loaded")

    print(f"📦 Loading YOLO: {YOLO_PATH}")
    yolo = YOLO(YOLO_PATH)
    print("✅ YOLO loaded")

    # Kích thước crop cần thiết (deduplicate)
    unique_sizes = sorted(set([APPROACH1_IMG_SIZE,
                               APPROACH2_IMG_SIZE,
                               APPROACH3_IMG_SIZE]))

    # ── Accumulators ─────────────────────────────────────────────
    a1_all_probs,  a1_all_labels  = [], []
    a1_video_preds, a1_video_labels = [], []
    a1_per_video = []

    a2_details = []
    a2_all_wprobs, a2_all_wlabels = [], []

    a3_all_probs,  a3_all_labels  = [], []
    a3_per_video = []

    # ═════════════════════════════════════════════════════════════
    # VÒNG LẶP CHÍNH — xử lý từng video
    # ═════════════════════════════════════════════════════════════
    print(f"\n🚀 Bắt đầu đánh giá {len(videos)} video ...\n" + "=" * 70)

    for idx, (vpath, label, display) in enumerate(videos):
        print(f"\n[{idx + 1}/{len(videos)}] {display}")

        # ── YOLO + crop 1 lần cho tất cả kích thước ─────────────
        results, duration = process_video_yolo(vpath, yolo, unique_sizes)
        print(f"  ⏱️  Duration: {duration:.1f}s")

        crops_96 = results[APPROACH1_IMG_SIZE]["track_crops"]
        times_96 = results[APPROACH1_IMG_SIZE]["track_times"]
        crops_64 = results[APPROACH2_IMG_SIZE]["track_crops"]
        times_64 = results[APPROACH2_IMG_SIZE]["track_times"]

        # ── Hướng 1: Sliding Window (96×96) ─────────────────────
        p1, l1, vpred1, vratio1 = evaluate_approach1(
            crops_96, label, model, device
        )
        a1_all_probs.extend(p1)
        a1_all_labels.extend(l1)
        a1_video_preds.append(vpred1)
        a1_video_labels.append(label)

        if p1:
            n_vwin   = sum(1 for p in p1 if p >= VIOLENCE_THRESHOLD)
            pred_s1  = "Violence" if vpred1 == 1 else "Non-Violence"
            label_s1 = "Violence" if label == 1  else "Non-Violence"
            icon1    = "✅" if vpred1 == label else "❌"
            a1_per_video.append({
                "name": display, "label": label,
                "pred": vpred1,
                "total_windows": len(p1),
                "violence_windows": n_vwin,
                "violence_window_ratio": vratio1,
            })
            print(f"  H1: {icon1} Pred={pred_s1} | Label={label_s1} "
                  f"| {n_vwin}/{len(p1)} windows violence "
                  f"({vratio1*100:.1f}%)")
        else:
            a1_per_video.append({
                "name": display, "label": label,
                "pred": 0,
                "total_windows": 0, "violence_windows": 0,
                "violence_window_ratio": 0.0,
                "note": "Không có track đủ dài",
            })
            print("  H1: Không có track đủ dài")

        # ── Hướng 2: Full Pipeline (64×64) ──────────────────────
        r2 = evaluate_approach2(crops_64, times_64, label, model, device)
        r2["name"] = display                 # gắn tên video vào kết quả
        a2_details.append(r2)
        a2_all_wprobs.extend(r2["window_probs"])
        a2_all_wlabels.extend(r2["window_labels"])

        pred_s  = "Violence" if r2["pred"] == 1 else "Non-Violence"
        label_s = "Violence" if label == 1      else "Non-Violence"
        icon    = "✅" if r2["pred"] == label else "❌"
        print(f"  H2: {icon} Pred={pred_s} | Label={label_s} "
              f"| ratio={r2['violence_ratio']:.3f}")

        # ── Hướng 3: No Sliding Window (96×96) ──────────────────
        p3, l3 = evaluate_approach3(crops_96, label, model, device)
        a3_all_probs.extend(p3)
        a3_all_labels.extend(l3)

        if p3:
            preds3   = [1 if p >= VIOLENCE_THRESHOLD else 0 for p in p3]
            correct3 = sum(a == b for a, b in zip(preds3, l3))
            acc3     = correct3 / len(preds3) * 100
            a3_per_video.append({
                "name": display, "label": label,
                "total_tracks": len(p3), "correct": correct3,
                "accuracy": acc3,
            })
            print(f"  H3: {len(p3)} tracks, acc={acc3:.1f}%")
        else:
            a3_per_video.append({
                "name": display, "label": label,
                "total_tracks": 0, "correct": 0, "accuracy": 0.0,
                "note": "Không có track đủ dài",
            })
            print("  H3: Không có track đủ dài")

        # Giải phóng bộ nhớ video này
        del results

    # ═════════════════════════════════════════════════════════════
    # TÍNH METRICS
    # ═════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 Tính toán metrics và vẽ biểu đồ ...\n")

    zero_m = {"accuracy": 0, "f1_violence": 0, "precision_violence": 0,
              "recall_violence": 0, "f1_non_violence": 0}

    # Hướng 1 — video-level metrics
    metrics1 = compute_metrics(a1_video_labels, a1_video_preds) \
               if a1_video_preds else zero_m.copy()

    # Hướng 2
    a2_vpreds  = [r["pred"]  for r in a2_details]
    a2_vlabels = [r["label"] for r in a2_details]
    metrics2 = compute_metrics(a2_vlabels, a2_vpreds) \
               if a2_details else zero_m.copy()

    # Hướng 3
    a3_preds = [1 if p >= VIOLENCE_THRESHOLD else 0 for p in a3_all_probs]
    metrics3 = compute_metrics(a3_all_labels, a3_preds) \
               if a3_all_probs else zero_m.copy()

    # ═════════════════════════════════════════════════════════════
    # VẼ BIỂU ĐỒ
    # ═════════════════════════════════════════════════════════════

    # ── Hướng 1 (video-level CM, window-level prob dist) ────────
    if a1_video_preds:
        plot_confusion_matrix(
            a1_video_labels, a1_video_preds,
            "Confusion Matrix — H1: Sliding Window (Video-level)",
            os.path.join(OUTPUT_DIR, "approach1_confusion_matrix.png"),
        )
    if a1_all_probs:
        plot_prob_distribution(
            a1_all_probs, a1_all_labels,
            "Phân phối P(Violence) — H1: Sliding Window",
            os.path.join(OUTPUT_DIR, "approach1_prob_distribution.png"),
        )
    if a1_per_video:
        plot_per_video_approach1(
            a1_per_video,
            os.path.join(OUTPUT_DIR, "approach1_per_video_results.png"),
        )

    # ── Hướng 2 ──────────────────────────────────────────────────
    if a2_details:
        plot_confusion_matrix(
            a2_vlabels, a2_vpreds,
            "Confusion Matrix — H2: Full Pipeline (Video-level)",
            os.path.join(OUTPUT_DIR, "approach2_confusion_matrix.png"),
        )
        plot_per_video_approach2(
            a2_details,
            os.path.join(OUTPUT_DIR, "approach2_per_video_results.png"),
        )
    if a2_all_wprobs:
        plot_prob_distribution(
            a2_all_wprobs, a2_all_wlabels,
            "Phân phối P(Violence) — H2: Full Pipeline",
            os.path.join(OUTPUT_DIR, "approach2_prob_distribution.png"),
        )

    # ── Hướng 3 ──────────────────────────────────────────────────
    if a3_all_probs:
        plot_confusion_matrix(
            a3_all_labels, a3_preds,
            "Confusion Matrix — H3: No Sliding Window (Track-level)",
            os.path.join(OUTPUT_DIR, "approach3_confusion_matrix.png"),
        )
        plot_prob_distribution(
            a3_all_probs, a3_all_labels,
            "Phân phối P(Violence) — H3: No Sliding Window",
            os.path.join(OUTPUT_DIR, "approach3_prob_distribution.png"),
        )
    if a3_per_video:
        plot_per_video_bar(
            a3_per_video,
            "Accuracy từng Video — H3: No Sliding Window",
            os.path.join(OUTPUT_DIR, "approach3_per_video_accuracy.png"),
        )

    # ── So sánh 3 hướng ─────────────────────────────────────────
    plot_comparison(
        {
            "H1: Sliding Window": metrics1,
            "H2: Full Pipeline":  metrics2,
            "H3: No Sliding":     metrics3,
        },
        os.path.join(OUTPUT_DIR, "comparison_summary.png"),
    )

    # ═════════════════════════════════════════════════════════════
    # XUẤT REPORT TEXT
    # ═════════════════════════════════════════════════════════════
    write_report(
        os.path.join(OUTPUT_DIR, "evaluation_report.txt"),
        videos,
        a1_all_probs, a1_all_labels, a1_per_video,
        a2_details, a2_all_wprobs, a2_all_wlabels,
        a3_all_probs, a3_all_labels, a3_per_video,
        metrics1, metrics2, metrics3,
        str(device),
    )

    # ═════════════════════════════════════════════════════════════
    # TÓM TẮT
    # ═════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("🎯 TÓM TẮT KẾT QUẢ:\n")
    print(f"  Hướng 1 (Sliding Window)  : "
          f"Acc = {metrics1['accuracy']:.2f}%  |  "
          f"F1(V) = {metrics1['f1_violence']:.2f}%")
    print(f"  Hướng 2 (Full Pipeline)   : "
          f"Acc = {metrics2['accuracy']:.2f}%  |  "
          f"F1(V) = {metrics2['f1_violence']:.2f}%")
    print(f"  Hướng 3 (No Sliding)      : "
          f"Acc = {metrics3['accuracy']:.2f}%  |  "
          f"F1(V) = {metrics3['f1_violence']:.2f}%")
    print(f"\n📁 Kết quả đã lưu tại: {OUTPUT_DIR}/")

    # Liệt kê file đã xuất
    print("\n📂 Các file đã xuất:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size  = os.path.getsize(fpath)
        print(f"  • {fname}  ({size / 1024:.0f} KB)")

    print("\n✅ HOÀN TẤT!")
