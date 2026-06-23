"""
violence_detector.py — Backend wrapper quanh model CNN-BiLSTM-Attention
Adapted từ front_end_test.py, thêm tính năng trả về timestamp bạo lực
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from train_ai import ImprovedViolenceModel

# ==========================================
# Hằng số — PHẢI khớp với front_end_test.py
# ==========================================
TARGET_FPS = 30
TARGET_IMG_SIZE = 64
WINDOW_LENGTH = 30
WINDOW_STEP = 5
VIOLENCE_THRESHOLD = 0.45
CONSECUTIVE_WINDOWS = 5   # Số windows liên tiếp vượt ngưỡng để cảnh báo (~2.5 giây)
YOLO_IMGSZ = 320
BATCH_INFERENCE = 8

# Singleton models
_model = None
_yolo = None
_device = None


def load_models(model_path: str, yolo_path: str):
    """Load model và YOLO một lần duy nhất khi khởi động server."""
    global _model, _yolo, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model = ImprovedViolenceModel(seq_length=WINDOW_LENGTH, img_size=TARGET_IMG_SIZE)
    state = torch.load(model_path, map_location=_device, weights_only=False)
    _model.load_state_dict(state)
    _model.to(_device)
    _model.eval()

    _yolo = YOLO(yolo_path)
    print(f"✅ Models loaded on {_device}")


def _normalize_frame_indices(original_fps, target_fps, total_frames):
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


def _merge_intervals(intervals: list, gap_sec: float = 1.0, padding: float = 0.5,
                     video_duration: float = 0.0) -> list:
    """Gộp các khoảng thời gian bạo lực gần nhau, thêm padding."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    cs, ce, cc = intervals[0]
    for s, e, c in intervals[1:]:
        if s <= ce + gap_sec:
            ce = max(ce, e)
            cc = max(cc, c)
        else:
            merged.append((cs, ce, cc))
            cs, ce, cc = s, e, c
    merged.append((cs, ce, cc))

    result = []
    for s, e, c in merged:
        result.append({
            'start_sec': round(max(0.0, s - padding), 2),
            'end_sec': round(min(video_duration, e + padding), 2),
            'confidence': round(c * 100, 1),
        })
    return result


def analyze_video(video_path: str) -> dict:
    """
    Phân tích video, trả về kết quả detection và timestamp bạo lực.

    Logic tính score theo time slot: dùng MAX thay vì MEAN.
    Lý do: Trong môi trường học đường, chỉ cần 1-2 người đánh nhau
    trong đám đông người xem là đã cần cảnh báo. Nếu dùng MEAN,
    xác suất bạo lực sẽ bị "loãng" bởi đám đông → bỏ sót bạo lực.
    Ví dụ: 2 người đánh (prob=0.85) + 15 người xem (prob=0.08) →
      MEAN = 0.17 (không phát hiện!) | MAX = 0.85 (phát hiện đúng!)

    Returns:
        {
            'is_violence': bool,
            'segments': [{'start_sec': float, 'end_sec': float, 'confidence': float}],
            'video_duration': float,
            'violence_ratio': float,   # tỉ lệ thời gian có bạo lực
            'max_violent_persons': int, # số người bạo lực tối đa cùng lúc
            'summary': str,
        }
    """
    if _model is None or _yolo is None:
        raise RuntimeError("Models chưa được load. Gọi load_models() trước.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / original_fps

    # Chuẩn hoá FPS
    if abs(original_fps - TARGET_FPS) > 0.5:
        resampled = _normalize_frame_indices(original_fps, TARGET_FPS, total_frames)
    else:
        resampled = list(range(total_frames))

    resampled_set = set(resampled)
    # Map: original frame idx → resampled time (seconds)
    orig_to_time = {oi: i / TARGET_FPS for i, oi in enumerate(resampled)}

    tracks_boxes = {}
    track_crops = {}     # {tid: [crop_chw, ...]}
    track_times = {}     # {tid: [time_sec, ...]}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = _yolo.track(
            frame, persist=True, conf=0.38, iou=0.55,
            classes=[0], tracker="bytetrack.yaml",
            verbose=False, imgsz=YOLO_IMGSZ,
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            for box, tid in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.id.int().cpu().numpy(),
            ):
                tid = int(tid)
                tracks_boxes.setdefault(tid, {})[frame_idx] = tuple(box.tolist())

        if frame_idx in resampled_set:
            t_sec = orig_to_time.get(frame_idx, frame_idx / original_fps)
            h_img, w_img = frame.shape[:2]

            for tid, fdict in tracks_boxes.items():
                if frame_idx not in fdict:
                    continue
                x1, y1, x2, y2 = fdict[frame_idx]
                bw, bh = x2 - x1, y2 - y1
                cx1 = max(0, int(x1 - .05 * bw))
                cy1 = max(0, int(y1 - .05 * bh))
                cx2 = min(w_img, int(x2 + .05 * bw))
                cy2 = min(h_img, int(y2 + .05 * bh))

                if cx2 > cx1 and cy2 > cy1:
                    crop = cv2.resize(frame[cy1:cy2, cx1:cx2],
                                      (TARGET_IMG_SIZE, TARGET_IMG_SIZE),
                                      interpolation=cv2.INTER_LINEAR)
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop = np.transpose(crop.astype(np.float32) / 255.0, (2, 0, 1))
                    track_crops.setdefault(tid, []).append(crop)
                    track_times.setdefault(tid, []).append(t_sec)

        frame_idx += 1

    cap.release()

    valid = {tid: c for tid, c in track_crops.items() if len(c) >= WINDOW_LENGTH}

    if not valid:
        return {
            'is_violence': False, 'segments': [],
            'video_duration': round(video_duration, 2),
            'violence_ratio': 0.0,
            'max_violent_persons': 0,
            'summary': 'Không phát hiện người / track đủ dài trong video.',
        }

    # ====== BƯỚC 1: Inference từng track — lấy prob cho mỗi window ======
    all_window_results = []  # [(center_time, prob, start_time, end_time)]

    for tid, crops in valid.items():
        arr = np.stack(crops, axis=0)
        times = track_times[tid]
        n = arr.shape[0]

        windows, starts = [], []
        for s in range(0, n - WINDOW_LENGTH + 1, WINDOW_STEP):
            windows.append(arr[s: s + WINDOW_LENGTH])
            starts.append(s)

        if not windows:
            continue

        probs_list = []
        _model.eval()
        with torch.no_grad():
            for bi in range(0, len(windows), BATCH_INFERENCE):
                batch = windows[bi: bi + BATCH_INFERENCE]
                x = torch.from_numpy(np.stack(batch)).to(_device, dtype=torch.float32)
                probs = _model.predict_proba(x)[:, 1].cpu().numpy()
                probs_list.extend(probs.tolist())

        for s, prob in zip(starts, probs_list):
            t_s = times[s] if s < len(times) else 0
            t_e = times[min(s + WINDOW_LENGTH - 1, len(times) - 1)]
            t_center = (t_s + t_e) / 2.0
            all_window_results.append((t_center, prob, t_s, t_e))

    if not all_window_results:
        return {
            'is_violence': False, 'segments': [],
            'video_duration': round(video_duration, 2),
            'violence_ratio': 0.0,
            'max_violent_persons': 0,
            'summary': 'Không trích xuất được cửa sổ nào từ các track.',
        }

    # ====== BƯỚC 2: Gom theo time slot — dùng MAX thay vì MEAN ======
    # Lý do dùng MAX: Trong bối cảnh trường học, chỉ cần 1 người
    # có hành vi bạo lực là đủ để cảnh báo. Nếu dùng MEAN, xác suất
    # sẽ bị "loãng" bởi đám đông người xem (thường chiếm đa số).
    # Ví dụ: 2 người đánh (0.85) + 15 người xem (0.08) →
    #   MEAN ≈ 0.17 (bỏ sót!) | MAX = 0.85 (phát hiện đúng!)
    #
    # Ngoài MAX score, còn đếm violent_person_count:
    # số người có prob ≥ VIOLENCE_THRESHOLD trong cùng slot.
    slot_duration = WINDOW_STEP / TARGET_FPS
    slot_data = defaultdict(list)

    for t_center, prob, t_s, t_e in all_window_results:
        slot_idx = int(t_center / slot_duration)
        slot_data[slot_idx].append((prob, t_s, t_e))

    max_slot = max(slot_data.keys())
    # timeline: [(slot_idx, max_score, t_s_min, t_e_max, n_violent_persons, n_total_persons)]
    timeline = []
    for s in range(max_slot + 1):
        if s in slot_data:
            entries = slot_data[s]
            # MAX: chỉ cần 1 người bạo lực là slot được tính là bạo lực
            max_score = float(np.max([p for p, _, _ in entries]))
            t_s_min = min(ts for _, ts, _ in entries)
            t_e_max = max(te for _, _, te in entries)
            # Đếm số người có prob ≥ ngưỡng trong slot này
            n_violent_persons = sum(1 for p, _, _ in entries if p >= VIOLENCE_THRESHOLD)
            n_total_persons = len(entries)
            timeline.append((s, max_score, t_s_min, t_e_max, n_violent_persons, n_total_persons))
        else:
            t = s * slot_duration
            timeline.append((s, 0.0, t, t + slot_duration, 0, 0))

    # ====== BƯỚC 3: Tìm ≥ CONSECUTIVE_WINDOWS slot liên tiếp vượt ngưỡng ======
    is_violence = False
    violent_segments = []
    run_start = None
    run_count = 0

    for i, (_, score, _, _, _, _) in enumerate(timeline):
        if score >= VIOLENCE_THRESHOLD:
            if run_count == 0:
                run_start = i
            run_count += 1
        else:
            if run_count >= CONSECUTIVE_WINDOWS:
                is_violence = True
                seg_start = timeline[run_start][2]
                seg_end = timeline[i - 1][3]
                seg_conf = max(timeline[j][1] for j in range(run_start, i))
                violent_segments.append((seg_start, seg_end, seg_conf))
            run_count = 0
            run_start = None

    # Xử lý trường hợp video kết thúc trong chuỗi bạo lực
    if run_count >= CONSECUTIVE_WINDOWS:
        is_violence = True
        seg_start = timeline[run_start][2]
        seg_end = timeline[-1][3]
        seg_conf = max(timeline[j][1] for j in range(run_start, len(timeline)))
        violent_segments.append((seg_start, seg_end, seg_conf))

    segments = _merge_intervals(violent_segments, video_duration=video_duration) \
        if is_violence else []

    # ====== THỐNG KÊ ======
    # violence_ratio: tỉ lệ time slot có bạo lực (dựa trên MAX score)
    n_violent_slots = sum(1 for _, score, _, _, _, _ in timeline
                          if score >= VIOLENCE_THRESHOLD)
    violence_ratio = n_violent_slots / max(len(timeline), 1)

    # max_violent_persons: số người bạo lực tối đa trong cùng 1 slot
    # (thống kê hữu ích: "tối đa X người đánh nhau cùng lúc")
    max_violent_persons = max(
        (n_v for _, _, _, _, n_v, _ in timeline), default=0
    )

    # peak_slot: slot có nhiều người bạo lực nhất → dùng cho summary
    peak_total_persons = max(
        (n_t for _, _, _, _, _, n_t in timeline if n_t > 0), default=0
    )

    summary = (
        f"Phát hiện bạo lực ({len(segments)} đoạn, "
        f"{n_violent_slots}/{len(timeline)} time-slot vượt ngưỡng, "
        f"tối đa {max_violent_persons} người bạo lực cùng lúc)"
        if is_violence else
        f"Không phát hiện bạo lực "
        f"(không có ≥{CONSECUTIVE_WINDOWS} time-slot liên tiếp vượt ngưỡng)"
    )

    return {
        'is_violence': is_violence,
        'segments': segments,
        'video_duration': round(video_duration, 2),
        'violence_ratio': round(violence_ratio, 3),
        'max_violent_persons': max_violent_persons,
        'summary': summary,
    }
