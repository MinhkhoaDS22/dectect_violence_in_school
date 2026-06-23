"""
Test với video từ Downloads
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from violence_detector import load_models, analyze_video

MODEL_PATH = r"D:\DATN\results\best_model.pth"
YOLO_PATH = r"D:\DATN\yolo11s.pt"

print("Loading models...")
load_models(MODEL_PATH, YOLO_PATH)

tests = [
    ("test.mp4", r"C:\Users\Admin\Downloads\test.mp4"),
    ("test1.mp4", r"C:\Users\Admin\Downloads\test1.mp4"),
]

for name, path in tests:
    print("\n" + "=" * 60)
    print(f"Analyzing: {name}")
    print("=" * 60)
    result = analyze_video(path)
    print(f"  is_violence   : {result['is_violence']}")
    print(f"  violence_ratio: {result['violence_ratio']}")
    print(f"  summary       : {result['summary']}")
    print(f"  duration      : {result['video_duration']}s")
    if result['segments']:
        for seg in result['segments']:
            print(f"  segment       : {seg['start_sec']}s - {seg['end_sec']}s (conf: {seg['confidence']}%)")
    else:
        print(f"  segments      : (none)")

print("\nDone!")
