import cv2
import os
from ultralytics import YOLO
from pathlib import Path
import torch

# --- BẢN VÁ LỖI PYTORCH 2.6+ ---
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

def export_cvat_xml_and_preview(video_path: str, output_xml_path: str, output_preview_path: str, model: YOLO, conf: float = 0.4, iou: float = 0.5):
    """
    Export annotations sang CVAT XML format VÀ xuất video preview để review offline.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không mở được video: {video_path}")
        return False

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Thiết lập VideoWriter cho file preview
    os.makedirs(os.path.dirname(output_preview_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_preview_path, fourcc, fps, (width, height))

    tracks = {}  

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            classes=[0],  # person
            tracker="bytetrack.yaml", 
            verbose=False
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, tid in zip(boxes, track_ids):
                tid = int(tid)  
                if tid not in tracks:
                    tracks[tid] = []

                x1, y1, x2, y2 = box.tolist()  
                tracks[tid].append({
                    'frame': frame_idx,
                    'xtl': x1, 'ytl': y1, 'xbr': x2, 'ybr': y2
                })

                # --- VẼ LÊN FRAME PREVIEW ---
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {tid}", (int(x1), max(0, int(y1) - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Ghi frame đã vẽ vào video preview
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release() # Đóng file video preview

    if not tracks:
        print(f"⚠️ Không có người nào được detect trong: {Path(video_path).name}")
        return False

    # --- GHI RA FILE XML CHUẨN CVAT ---
    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<annotations>\n')
        f.write('  <version>1.1</version>\n')
        f.write('  <meta>\n')
        f.write('    <task>\n')
        f.write(f'      <size>{num_frames}</size>\n')
        f.write('    </task>\n')
        f.write('  </meta>\n')
        
        for track_id, boxes in tracks.items():
            f.write(f'  <track id="{track_id}" label="person">\n')
            
            for box in boxes:
                f.write(f'    <box frame="{box["frame"]}" outside="0" occluded="0" keyframe="1" ')
                f.write(f'xtl="{box["xtl"]:.2f}" ytl="{box["ytl"]:.2f}" xbr="{box["xbr"]:.2f}" ybr="{box["ybr"]:.2f}">\n')
                f.write('    </box>\n')
            
            last_box = boxes[-1]
            last_frame = last_box["frame"]
            
            if last_frame < num_frames - 1:
                f.write(f'    <box frame="{last_frame + 1}" outside="1" occluded="0" keyframe="1" ')
                f.write(f'xtl="{last_box["xtl"]:.2f}" ytl="{last_box["ytl"]:.2f}" xbr="{last_box["xbr"]:.2f}" ybr="{last_box["ybr"]:.2f}">\n')
                f.write('    </box>\n')
                
            f.write('  </track>\n')
        f.write('</annotations>\n')
        
    return True

def process_all_videos(data_dir: str, xml_dir: str, preview_dir: str):
    print("🚀 Đang khởi tạo mô hình YOLO...")
    model = YOLO('yolo11s.pt') 
    
    categories = ['violence', 'non_violence']
    
    for category in categories:
        input_folder = os.path.join(data_dir, category)
        xml_folder = os.path.join(xml_dir, category)
        preview_folder = os.path.join(preview_dir, category)
        
        if not os.path.exists(input_folder):
            print(f"⚠️ Bỏ qua vì không thấy thư mục: {input_folder}")
            continue
            
        videos = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
        videos.sort() 
        total_videos = len(videos)
        
        print(f"\n--- Bắt đầu xử lý thư mục: {category.upper()} ({total_videos} video) ---")
        
        for idx, video_name in enumerate(videos):
            video_path = os.path.join(input_folder, video_name)
            
            base_name = os.path.splitext(video_name)[0]
            output_xml_path = os.path.join(xml_folder, base_name + '.xml')
            output_preview_path = os.path.join(preview_folder, base_name + '_preview.mp4')
            
            # Tính năng Skip: Nếu CẢ file XML VÀ PREVIEW đã tồn tại thì bỏ qua
            if os.path.exists(output_xml_path) and os.path.exists(output_preview_path):
                print(f"⏭️ [{idx+1}/{total_videos}] Đã tồn tại {base_name}. Đang bỏ qua...")
                continue
                
            print(f"🔄 [{idx+1}/{total_videos}] Đang chạy tracking: {video_name}...")
            export_cvat_xml_and_preview(video_path, output_xml_path, output_preview_path, model, conf=0.38, iou=0.55)

if __name__ == "__main__":
    DATA_DIRECTORY = "data"        
    XML_OUTPUT_DIRECTORY = "labels"  
    PREVIEW_OUTPUT_DIRECTORY = "previews" # Thư mục chứa video nháp để em xem lại
    
    process_all_videos(DATA_DIRECTORY, XML_OUTPUT_DIRECTORY, PREVIEW_OUTPUT_DIRECTORY)
    print("\n✅ ĐÃ HOÀN TẤT! Em hãy vào thư mục 'previews' để xem video kết quả nhé.")