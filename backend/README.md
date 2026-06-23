# Hướng dẫn chạy SafeWatch

## Bước 1 — Cài đặt Backend

```powershell
# Vào thư mục backend
cd d:\DATN\backend

# Tạo file .env từ template
copy .env.example .env
# → Mở .env và điền GMAIL_APP_PASSWORD + TELEGRAM_BOT_TOKEN

# Cài thư viện (nên dùng virtual env)
pip install -r requirements.txt

# Cài thêm requests (dùng bởi notifier.py)
pip install requests
```

## Bước 2 — Cấu hình .env

Mở `d:\DATN\backend\.env` và điền:

```
GMAIL_SENDER=email_cua_ban@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   # App Password từ Google

TELEGRAM_BOT_TOKEN=123456:ABCdef...       # Token từ @BotFather
```

**Lấy Telegram Chat ID:**
1. Tìm bot của bạn trên Telegram
2. Nhắn `/start`
3. Vào: `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Copy `chat.id` từ response

## Bước 3 — Chạy Backend

```powershell
cd d:\DATN\backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend sẽ:
- Load model `best_model_v2.pth` (8MB)
- Load YOLO `yolo11s.pt` (19MB)
- Khởi động API tại `http://localhost:8000`

## Bước 4 — Chạy Flutter Web

```powershell
cd d:\DATN\violence_app
flutter run -d chrome
```

App mở trên Chrome tại `http://localhost:XXXX`

## Flow sử dụng

1. **Onboarding**: Nhập email và/hoặc Telegram Chat ID
2. **Kiểm tra kết nối**: Bấm "Kiểm tra kết nối" → đợi backend response
3. **Monitor tab**: Kéo thả hoặc chọn video → bấm Upload
4. Backend phân tích (~30s-5min tuỳ video)
5. Kết quả hiển thị:
   - Nếu bình thường: ✅
   - Nếu bạo lực: 🚨 + thông báo tới email/Telegram kèm clip

## Lưu ý

- Video càng dài → phân tích càng lâu
- Clip sẽ được lưu tại `backend/clips/`
- Lịch sử cảnh báo lưu tại `backend/alerts.json`
- Gmail: Phải bật 2FA và tạo App Password
- Telegram: Người dùng phải nhắn `/start` cho bot TRƯỚC khi nhận tin nhắn
