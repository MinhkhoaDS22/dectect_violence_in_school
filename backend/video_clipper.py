"""
video_clipper.py — Cắt video theo timestamp dùng OpenCV
"""

import cv2
import os
from datetime import datetime


def clip_video(video_path: str, start_sec: float, end_sec: float,
               output_path: str) -> str:
    """
    Cắt video từ start_sec đến end_sec, thêm watermark cảnh báo.

    Returns:
        Đường dẫn file clip đã tạo
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total_frames - 1, int(end_sec * fps))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # --- Watermark: dải đỏ phía trên ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 38), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        t = start_sec + i / fps
        label = f"  VIOLENCE DETECTED  |  t = {t:.1f}s  |  SafeWatch"
        cv2.putText(frame, label, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    return output_path


def clip_all_segments(video_path: str, segments: list, clips_dir: str) -> list:
    """
    Cắt tất cả đoạn bạo lực trong video.

    Args:
        segments: [{'start_sec': float, 'end_sec': float, 'confidence': float}]
        clips_dir: Thư mục lưu clip

    Returns:
        List đường dẫn các clip đã tạo
    """
    os.makedirs(clips_dir, exist_ok=True)
    clip_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, seg in enumerate(segments):
        clip_name = f"clip_seg{i + 1}_{ts}.mp4"
        clip_path = os.path.join(clips_dir, clip_name)
        try:
            path = clip_video(video_path, seg['start_sec'], seg['end_sec'], clip_path)
            clip_paths.append(path)
        except Exception as e:
            print(f"❌ Lỗi cắt segment {i + 1}: {e}")

    return clip_paths
