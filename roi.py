import cv2
import torch
import os
from tqdm import tqdm
from datetime import datetime

INPUT_DIR = r"D:\DIF_Videos"  
OUTPUT_DIR = r"D:\DIF_Videos_ROI"  

DRUNK_INPUT = os.path.join(INPUT_DIR, "drunk")
SOBER_INPUT = os.path.join(INPUT_DIR, "sober")

DRUNK_OUTPUT = os.path.join(OUTPUT_DIR, "drunk")
SOBER_OUTPUT = os.path.join(OUTPUT_DIR, "sober")

ROI_SIZE = 256

#O/p directory
os.makedirs(DRUNK_OUTPUT, exist_ok=True)
os.makedirs(SOBER_OUTPUT, exist_ok=True)

print(f"Input directory:  {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}\n")

#https://github.com/ultralytics/yolov5
#https://pytorch.org/hub/ultralytics_yolov5/
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.5  # Confidence threshold
print("YOLOv5 model loaded\n")

# extraction fucntion
def extract_roi_from_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
            return {
                'video': os.path.basename(input_video_path),
                'status': 'FAILED',
                'error': 'Could not open video',
                'frames_processed': 0,
                'frames_saved': 0
            }
    
    # video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
