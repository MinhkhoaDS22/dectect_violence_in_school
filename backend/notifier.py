"""
notifier.py — Gửi cảnh báo qua Gmail (SMTP) và Telegram Bot
"""

import os
import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import Optional


# ==========================================
# GMAIL
# ==========================================
def _build_email_html(detection: dict, video_filename: str) -> str:
    segments_rows = ""
    for i, seg in enumerate(detection.get('segments', [])):
        segments_rows += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #2a2a2e;">Đoạn {i+1}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #2a2a2e;">
            {seg['start_sec']:.1f}s → {seg['end_sec']:.1f}s
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #2a2a2e;color:#ff453a;font-weight:700;">
            {seg['confidence']}%
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0d0d0f;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#fff;">
  <div style="max-width:600px;margin:0 auto;padding:24px;">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#ff3b30 0%,#ff6b35 100%);
                border-radius:16px;padding:36px 32px;text-align:center;margin-bottom:20px;">
      <div style="font-size:48px;margin-bottom:12px;">🚨</div>
      <h1 style="margin:0 0 8px;font-size:26px;font-weight:800;letter-spacing:-0.5px;">
        PHÁT HIỆN BẠO LỰC
      </h1>
      <p style="margin:0;opacity:.88;font-size:15px;">
        SafeWatch phát hiện hành vi bất thường trong video của bạn
      </p>
    </div>

    <!-- Video info -->
    <div style="background:#1c1c1e;border-radius:12px;padding:20px 24px;margin-bottom:16px;">
      <h2 style="margin:0 0 14px;font-size:16px;color:#0a84ff;text-transform:uppercase;letter-spacing:.5px;">
        📹 Thông tin Video
      </h2>
      <p style="margin:4px 0;font-size:14px;color:#ebebf5cc;">
        <strong style="color:#fff;">File:</strong> {video_filename}
      </p>
      <p style="margin:4px 0;font-size:14px;color:#ebebf5cc;">
        <strong style="color:#fff;">Thời lượng:</strong> {detection.get('video_duration', 0):.1f}s
      </p>
      <p style="margin:4px 0;font-size:14px;color:#ebebf5cc;">
        <strong style="color:#fff;">Tỷ lệ track bạo lực:</strong>
        {detection.get('violence_ratio', 0)*100:.1f}%
      </p>
      <p style="margin:4px 0;font-size:14px;color:#ebebf5cc;">
        <strong style="color:#fff;">Phân tích:</strong> {detection.get('summary', '')}
      </p>
    </div>

    <!-- Segments table -->
    <div style="background:#1c1c1e;border-radius:12px;overflow:hidden;margin-bottom:16px;">
      <div style="padding:16px 24px 12px;border-bottom:1px solid #2a2a2e;">
        <h2 style="margin:0;font-size:16px;color:#ff453a;text-transform:uppercase;letter-spacing:.5px;">
          ⏱ Đoạn Bạo Lực
        </h2>
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
          <tr style="background:#2a2a2e;color:#ebebf599;">
            <th style="padding:10px 12px;text-align:left;font-weight:600;">Đoạn</th>
            <th style="padding:10px 12px;text-align:left;font-weight:600;">Thời điểm</th>
            <th style="padding:10px 12px;text-align:left;font-weight:600;">Độ tin cậy</th>
          </tr>
        </thead>
        <tbody>{segments_rows}</tbody>
      </table>
    </div>

    <!-- Clip note -->
    <div style="background:#0a84ff18;border:1px solid #0a84ff44;border-radius:10px;
                padding:14px 20px;margin-bottom:24px;">
      <p style="margin:0;font-size:13px;color:#0a84ff;">
        📎 Clip video chỉ chứa đoạn bạo lực được đính kèm trong email này.
      </p>
    </div>

    <!-- Footer -->
    <p style="text-align:center;color:#ebebf540;font-size:12px;margin:0;">
      SafeWatch Violence Detection System •
      {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
    </p>
  </div>
</body>
</html>"""


def send_gmail(to_email: str, detection: dict, video_filename: str,
               clip_paths: list) -> bool:
    """Gửi email cảnh báo kèm clip bạo lực."""
    sender = os.getenv("GMAIL_SENDER", "").strip()
    # App Password có thể có dấu cách (vd: "ilwd tdkf wceb dipq") → strip hết
    password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()

    if not sender or not password:
        print("⚠️  Gmail chưa cấu hình: kiểm tra GMAIL_SENDER và GMAIL_APP_PASSWORD trong .env")
        return False

    print(f"📧 Đang gửi Gmail tới {to_email}...")
    print(f"   Sender : {sender}")
    print(f"   PwdLen : {len(password)} ký tự")

    try:
        msg = MIMEMultipart('mixed')
        # KHÔNG dùng emoji trong From header — một số mail server reject
        msg['From'] = f"SafeWatch Alert <{sender}>"
        msg['To'] = to_email
        msg['Subject'] = (
            f"[SafeWatch] Phat hien bao luc - "
            f"{datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )

        html = _build_email_html(detection, video_filename)
        msg.attach(MIMEText(html, 'html', 'utf-8'))

        # Đính kèm clip
        attached = 0
        for clip_path in clip_paths:
            if not os.path.exists(clip_path):
                print(f"   ⚠️  Clip không tồn tại: {clip_path}")
                continue
            with open(clip_path, 'rb') as f:
                part = MIMEBase('video', 'mp4')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment',
                            filename=os.path.basename(clip_path))
            msg.attach(part)
            attached += 1

        # Dùng SMTP + STARTTLS port 587 (ổn định hơn SMTP_SSL 465 trên Windows)
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, to_email, msg.as_string())

        print(f"✅ Gmail gửi thành công tới {to_email} ({attached} clip đính kèm)")
        return True

    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Gmail AUTH ERROR: Sai App Password hoặc chưa bật 2FA")
        print(f"   Chi tiết: {e}")
        return False
    except smtplib.SMTPException as e:
        print(f"❌ Gmail SMTP ERROR: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Gmail UNKNOWN ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ==========================================
# TELEGRAM
# ==========================================
def _phone_to_hint(phone: str) -> str:
    """Gợi ý người dùng tìm chat_id."""
    return phone


def send_telegram(chat_id_or_phone: str, detection: dict,
                  video_filename: str, clip_paths: list) -> bool:
    """
    Gửi cảnh báo + clip qua Telegram Bot.

    chat_id_or_phone: Telegram chat_id (số nguyên dạng chuỗi).
    Người dùng phải nhắn /start cho bot trước để lấy chat_id.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("⚠️  Telegram bot token chưa cấu hình (.env)")
        return False

    try:
        import requests

        base = f"https://api.telegram.org/bot{token}"

        segs = detection.get('segments', [])
        seg_lines = "\n".join(
            f"  • Đoạn {i+1}: {s['start_sec']:.1f}s → {s['end_sec']:.1f}s "
            f"(độ tin cậy: {s['confidence']}%)"
            for i, s in enumerate(segs)
        )

        text = (
            f"🚨 *PHÁT HIỆN BẠO LỰC* 🚨\n\n"
            f"📹 *File:* `{video_filename}`\n"
            f"⏱ *Thời lượng:* {detection.get('video_duration', 0):.1f}s\n"
            f"📊 *Tỷ lệ:* {detection.get('violence_ratio', 0)*100:.1f}%\n\n"
            f"⏱ *Các đoạn bạo lực:*\n{seg_lines}\n\n"
            f"📎 Clip chi tiết được gửi ngay bên dưới.\n"
            f"_SafeWatch • {datetime.now().strftime('%d/%m/%Y %H:%M')}_"
        )

        # Gửi text
        requests.post(f"{base}/sendMessage", json={
            'chat_id': chat_id_or_phone,
            'text': text,
            'parse_mode': 'Markdown',
        }, timeout=10)

        # Gửi từng clip
        for clip_path in clip_paths:
            if not os.path.exists(clip_path):
                continue
            with open(clip_path, 'rb') as f:
                requests.post(f"{base}/sendVideo",
                              data={'chat_id': chat_id_or_phone,
                                    'caption': f"🎬 Clip bạo lực — {os.path.basename(clip_path)}"},
                              files={'video': f},
                              timeout=60)

        print(f"✅ Telegram gửi tới chat_id={chat_id_or_phone} ({len(clip_paths)} clip)")
        return True

    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False
