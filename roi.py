import cv2
import torch
import os
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

    try:
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            return {
                'video': os.path.basename(input_video_path),
                'status': 'FAILED',
                'error': 'Could not open video',
                'frames_processed': 0,
                'frames_saved': 0
            }
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        out = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        frames_processed = 0
        frames_saved = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frames_processed += 1
            
            # Detect faces using YOLOv5
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()
            
            if len(detections) > 0:
                # Get the first (largest) face detection
                x_min, y_min, x_max, y_max, conf, cls = detections[0]
                
                # Convert to int
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                
                # Ensure coordinates are within frame bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(width, x_max)
                y_max = min(height, y_max)
                
                # Extract face ROI
                face_roi = frame[y_min:y_max, x_min:x_max]
                
                # Skip if ROI is too small
                if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                    continue
                
                # Resize to standard size
                face_roi_resized = cv2.resize(face_roi, (ROI_SIZE, ROI_SIZE))
                
                # Initialize video writer on first face detection
                if out is None:
                    out = cv2.VideoWriter(
                        output_video_path, 
                        fourcc, 
                        fps, 
                        (ROI_SIZE, ROI_SIZE)
                    )
                
                # Write frame to output video
                out.write(face_roi_resized)
                frames_saved += 1
        
        # Release resources
        cap.release()
        if out is not None:
            out.release()
        
        # Check if output file was created successfully
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 1000:
            detection_rate = (frames_saved / frames_processed * 100) if frames_processed > 0 else 0
            
            return {
                'video': os.path.basename(input_video_path),
                'status': 'SUCCESS',
                'frames_processed': frames_processed,
                'frames_saved': frames_saved,
                'detection_rate': f"{detection_rate:.1f}%"
            }
        else:
            return {
                'video': os.path.basename(input_video_path),
                'status': 'FAILED',
                'error': 'No frames with detected face',
                'frames_processed': frames_processed,
                'frames_saved': frames_saved
            }
    
    except Exception as e:
        return {
            'video': os.path.basename(input_video_path),
            'status': 'FAILED',
            'error': str(e)[:100],
            'frames_processed': 0,
            'frames_saved': 0
        }