import cv2
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes_by_frame = {}
    
    for track in root.findall('track'):
        track_id = track.get('id')
        for box in track.findall('box'):
            if box.get('outside') == '1':
                continue
            frame_idx = int(box.get('frame'))
            x1, y1, x2, y2 = float(box.get('xtl')), float(box.get('ytl')), float(box.get('xbr')), float(box.get('ybr'))
            
            if frame_idx not in boxes_by_frame:
                boxes_by_frame[frame_idx] = []
            boxes_by_frame[frame_idx].append((track_id, x1, y1, x2, y2))
    return boxes_by_frame

def find_original_video(video_name, data_dir="data"):
    for category in ['violence', 'non_violence']:
        potential_path = os.path.join(data_dir, category, video_name)
        if os.path.exists(potential_path):
            return potential_path
    return None

def create_preview_from_xml(xml_path, data_dir="data", output_dir="fixed_previews"):
    base_name = Path(xml_path).stem
    
    video_path = find_original_video(base_name + '.mp4', data_dir)
    if not video_path:
        video_path = find_original_video(base_name + '.avi', data_dir)
        
    if not video_path:
        print(f"❌ Lỗi: Không tìm thấy video gốc cho file '{base_name}' trong thư mục '{data_dir}'")
        return

    boxes_by_frame = parse_cvat_xml(xml_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể mở video: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    os.makedirs(output_dir, exist_ok=True)
    output_preview_path = os.path.join(output_dir, base_name + '_fixed.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_preview_path, fourcc, fps, (width, height))
    
    print(f"🔄 Đang vẽ preview VÀNG cho video: {base_name}...")
    
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        if frame_idx in boxes_by_frame:
            for box_data in boxes_by_frame[frame_idx]:
                track_id, x1, y1, x2, y2 = box_data
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Vẽ box màu Vàng
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, max(0, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
        out.write(frame)
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"✅ Xong! Đã lưu video tại: {output_preview_path}")

# ==========================================
# CHỈ CHẠY 1 FILE DUY NHẤT
# ==========================================
if __name__ == "__main__":
    DATA_DIR = "data"                   
    OUTPUT_PREVIEW_DIR = "fixed_previews" 
    
    # 👇 ĐỔI TÊN FILE XML EM MUỐN TEST Ở DÒNG NÀY 👇
    TARGET_XML = "fix_labels/violence/v_001.xml"  
    
    if not os.path.exists(TARGET_XML):
        print(f"⚠️ Không tìm thấy file: {TARGET_XML}. Em kiểm tra lại xem đã copy file vào thư mục fix_labels chưa nhé.")
    else:
        create_preview_from_xml(TARGET_XML, DATA_DIR, OUTPUT_PREVIEW_DIR)
        print("\n🎉 Mời em vào thư mục 'fixed_previews' để nghiệm thu thành quả!")