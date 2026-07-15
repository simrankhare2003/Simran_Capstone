import torch
import cv2
import numpy as np
import os
from pathlib import Path
import csv
from scipy import signal
import sys

sys.path.insert(0, r'D:\rPPG-Toolbox\neural_methods\model')
from RhythmFormer import RhythmFormer

# CONFIGURATION

ROI_VIDEO_DIR = r"D:\DIF_Videos_ROI"
OUTPUT_DIR = r"D:\rPPG_Results"
WAVEFORM_DIR = os.path.join(OUTPUT_DIR, "waveforms")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "rppg_features.csv")

MODEL_PATH = r"D:\rPPG-Toolbox\final_model_release\PURE_RhythmFormer.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model parameters
FRAME_WINDOW = 160
IMAGE_SIZE = (128, 128)
FPS = 30

# Create output directories
os.makedirs(WAVEFORM_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD MODEL

print("Loading RhythmFormer model...")
model = RhythmFormer(
    dim=64,
    frame=FRAME_WINDOW,
    image_size=(FRAME_WINDOW, IMAGE_SIZE[0], IMAGE_SIZE[1]),
    in_chans=64,
    head_dim=16,
    stage_n=3,
    embed_dim=[64, 64, 64],
    mlp_ratios=[1.5, 1.5, 1.5],
    depth=[2, 2, 2],
    t_patchs=(2, 4, 8),
    topks=(40, 40, 40),
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

new_checkpoint = {}
for key, value in checkpoint.items():
    if key.startswith('module.'):
        new_key = key[7:]  # Remove 'module.' prefix
        new_checkpoint[new_key] = value
    else:
        new_checkpoint[key] = value

model.load_state_dict(new_checkpoint)

model.to(DEVICE)
model.eval()
print(f"Model loaded\n")

def load_video_frames(video_path, num_frames=FRAME_WINDOW):

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to match model's expected input resolution
        frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    
    cap.release()
    
    
    if len(frames) < num_frames:
        if len(frames) == 0:
            return None
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames.append(last_frame)
    
    return np.array(frames[:num_frames])  # (D, H, W, C)



def extract_heart_rate(rppg_signal, fps=FPS, method='fft'):
    
    # Detrend signal
    signal_detrended = signal.detrend(rppg_signal)
    
    if method == 'fft':
        # FFT to get frequency domain
        fft = np.fft.fft(signal_detrended)
        freqs = np.fft.fftfreq(len(signal_detrended), 1/fps)
        
        # Only positive frequencies, HR range 40-200 bpm
        valid_freqs = freqs[(freqs >= 40/60) & (freqs <= 200/60)]
        valid_fft = fft[(freqs >= 40/60) & (freqs <= 200/60)]
        
        if len(valid_fft) > 0:
            hr_freq = valid_freqs[np.argmax(np.abs(valid_fft))]
            hr = hr_freq * 60  # Convert to BPM
        else:
            hr = 0
    else:
        hr = 0
    
    return max(0, hr)  # Ensure non-negative

def extract_hrv(rppg_signal, fps=FPS):
   
    # Find peaks (heart beats)
    signal_detrended = signal.detrend(rppg_signal)
    peaks, _ = signal.find_peaks(signal_detrended, distance=fps//2)  # Min distance between peaks
    
    if len(peaks) < 2:
        return 0
    
    # Calculate time between beats in milliseconds
    nn_intervals = np.diff(peaks) / fps * 1000
    
    # SDNN (Standard Deviation of NN intervals)
    hrv_sdnn = np.std(nn_intervals)
    
    # Root Mean Square of Successive Differences
    successive_diffs = np.diff(nn_intervals)
    hrv_rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    
    return hrv_sdnn, hrv_rmssd

def process_video(video_path, category):
    
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    
    print(f"Processing: {video_id} ({category})")
    
    try:
        # Load video frames
        frames = load_video_frames(video_path, FRAME_WINDOW)
        
        if frames is None or len(frames) == 0:
            print(f" Could not load frames")
            return None
        
        # Convert to tensor (N, D, C, H, W) format
        # frames is (D, H, W, C), need to convert to (N, D, C, H, W)
        frames_tensor = torch.from_numpy(frames).float()  # (D, H, W, C)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, C, D, H, W)
        frames_tensor = frames_tensor / 255.0  # Normalize to [0, 1]
        frames_tensor = frames_tensor.to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            print("frames_tensor shape:", frames_tensor.shape)
            rppg_signal = model(frames_tensor)  # (1, D)
            rppg_signal = rppg_signal.squeeze(0).cpu().numpy()  # (D,)
        
        # Extract features
        hr = extract_heart_rate(rppg_signal)
        hrv_data = extract_hrv(rppg_signal)
        
        if hrv_data is None:
            hrv_sdnn, hrv_rmssd = 0, 0
        else:
            hrv_sdnn, hrv_rmssd = hrv_data
        
        # Signal statistics
        signal_mean = float(np.mean(rppg_signal))
        signal_std = float(np.std(rppg_signal))
        signal_min = float(np.min(rppg_signal))
        signal_max = float(np.max(rppg_signal))
        
        # Save waveform
        waveform_path = os.path.join(WAVEFORM_DIR, f"{video_id}_waveform.npy")
        np.save(waveform_path, rppg_signal)
        
        print(f"  HR: {hr:.1f} bpm, HRV_SDNN: {hrv_sdnn:.2f}, HRV_RMSSD: {hrv_rmssd:.2f}")
        
        return {
            'video_id': video_id,
            'category': category,
            'heart_rate': round(hr, 2),
            'hrv_sdnn': round(hrv_sdnn, 2),
            'hrv_rmssd': round(hrv_rmssd, 2),
            'signal_mean': round(signal_mean, 4),
            'signal_std': round(signal_std, 4),
            'signal_min': round(signal_min, 4),
            'signal_max': round(signal_max, 4),
            'waveform_file': waveform_path
        }
  
    except Exception as e:
        import traceback
        print(f"  Error: {str(e)}")
        traceback.print_exc()
        return None

all_results = []

# Process drunk videos
drunk_dir = os.path.join(ROI_VIDEO_DIR, "drunk")
drunk_videos = [f for f in os.listdir(drunk_dir) if f.endswith('.mp4')]

print(f"Processing {len(drunk_videos)} DRUNK videos...\n")
for video_file in drunk_videos:
    video_path = os.path.join(drunk_dir, video_file)
    result = process_video(video_path, 'drunk')
    if result:
        all_results.append(result)

print()

# Process sober videos
sober_dir = os.path.join(ROI_VIDEO_DIR, "sober")
sober_videos = [f for f in os.listdir(sober_dir) if f.endswith('.mp4')]

print(f"Processing {len(sober_videos)} SOBER videos...\n")
for video_file in sober_videos:
    video_path = os.path.join(sober_dir, video_file)
    result = process_video(video_path, 'sober')
    if result:
        all_results.append(result)

# Save to CSV
if all_results:
    with open(CSV_OUTPUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"CSV saved: {CSV_OUTPUT}")
    print(f" Waveforms saved to: {WAVEFORM_DIR}")
    print(f" Total videos processed: {len(all_results)}\n")
else:
    print("No results to save")

