import os

def rename_dataset_videos(data_dir):
    # Cấu hình thư mục và tiền tố tương ứng
    # non_violence -> tiền tố 'nv_'
    # violence -> tiền tố 'v_'
    categories = {
        'non_violence': 'nv_',
        'violence': 'v_'
    }
    
    for category, prefix in categories.items():
        folder_path = os.path.join(data_dir, category)
        
        # Kiểm tra xem thư mục có tồn tại không
        if not os.path.exists(folder_path):
            print(f"Cảnh báo: Không tìm thấy thư mục {folder_path}")
            continue
            
        # Lọc ra các file video và sắp xếp chúng theo thứ tự tên hiện tại
        files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
        files.sort()
        
        print(f"\n--- Đang đổi tên {len(files)} video trong thư mục '{category}' ---")
        
        for index, filename in enumerate(files):
            # Lấy phần đuôi file (ví dụ: .mp4)
            file_extension = os.path.splitext(filename)[1]
            
            # Tạo tên mới định dạng 3 chữ số: nv_001.mp4, nv_002.mp4...
            new_name = f"{prefix}{index + 1:03d}{file_extension}"
            
            old_filepath = os.path.join(folder_path, filename)
            new_filepath = os.path.join(folder_path, new_name)
            
            # Đổi tên file
            os.rename(old_filepath, new_filepath)
            print(f"Đã đổi: {filename} -> {new_name}")

if __name__ == "__main__":
    # Đảm bảo thư mục 'data' nằm cùng cấp với file code này
    DATA_DIR = "data" 
    
    print("Bắt đầu chuẩn hóa tên file dataset...")
    rename_dataset_videos(DATA_DIR)
    print("\nĐã dọn dẹp và đổi tên xong toàn bộ dataset!")