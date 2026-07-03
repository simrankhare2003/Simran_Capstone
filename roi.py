import cv2
import os

INPUT_DIR = r"D:\DIF_Videos"  
OUTPUT_DIR = r"D:\DIF_Videos_ROI"  

DRUNK_INPUT = os.path.join(INPUT_DIR, "drunk")
SOBER_INPUT = os.path.join(INPUT_DIR, "sober")

DRUNK_OUTPUT = os.path.join(OUTPUT_DIR, "drunk")
SOBER_OUTPUT = os.path.join(OUTPUT_DIR, "sober")

ROI_SIZE = 256  

# output directories
os.makedirs(DRUNK_OUTPUT, exist_ok=True)
os.makedirs(SOBER_OUTPUT, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
print("Haar Cascade loaded\n")

# ROI EXTRACTION FUNCTION

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
        
        #video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        #video writer
        out = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        frames_processed = 0
        frames_saved = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frames_processed += 1
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            
            if len(faces) > 0:
                # Get face
                (x, y, w, h) = faces[0]
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                x_max = min(width, x + w)
                y_max = min(height, y + h)
                
                # Extract face ROI
                face_roi = frame[y:y_max, x:x_max]
                
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
        
        cap.release()
        if out is not None:
            out.release()
        
        # Check if output file was created
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

# PROCESS ALL VIDEOS

def process_category(input_dir, output_dir, category_name):
    """Process all videos in a category (silent)"""
    
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    results = []
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, video_file)
        
        result = extract_roi_from_video(input_path, output_path)
        results.append(result)
    
    return results

print("Starting ROI extraction...")

input("Press enter to start...")

# Process videos
drunk_results = process_category(DRUNK_INPUT, DRUNK_OUTPUT, "drunk")
sober_results = process_category(SOBER_INPUT, SOBER_OUTPUT, "sober")

print("ROI EXTRACTION COMPLETE")

# drunk_success = sum(1 for r in drunk_results if r['status'] == 'SUCCESS')
# sober_success = sum(1 for r in sober_results if r['status'] == 'SUCCESS')

# total_drunk_frames = sum(r.get('frames_processed', 0) for r in drunk_results)
# total_drunk_saved = sum(r.get('frames_saved', 0) for r in drunk_results)

# total_sober_frames = sum(r.get('frames_processed', 0) for r in sober_results)
# total_sober_saved = sum(r.get('frames_saved', 0) for r in sober_results)

# print(f"DRUNK: {drunk_success}/{len(drunk_results)} processed | {total_drunk_saved:,} frames saved")
# print(f"SOBER: {sober_success}/{len(sober_results)} processed | {total_sober_saved:,} frames saved\n")

# total_success = drunk_success + sober_success
# total_videos = len(drunk_results) + len(sober_results)

# print(f"Overall: {total_success}/{total_videos} videos processed successfully")
# print(f"Output: {OUTPUT_DIR}\n")
