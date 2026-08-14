import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
BASE_PATH = r"C:\Users\simrankhare\Desktop\Simran\college\Capstone\dataset\PUCV-DTF"
SOBER_DIR = os.path.join(BASE_PATH, "sobrios_v2")
DRUNK_DIR = os.path.join(BASE_PATH, "4_cerveza_v2")

OUTPUT_DIR = os.path.join(BASE_PATH, "Thermal_Analysis_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_person_folders(directory):
    return [d for d in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, d))]

def get_person_name_prefix(folder_name):
    parts = folder_name.rsplit('_', 1)  # Split from right, only once
    return parts[0]

def match_folders(sober_folders, drunk_folders):
    matches = {}
    
    # Create dicts by prefix
    sober_dict = {get_person_name_prefix(f): f for f in sober_folders}
    drunk_dict = {get_person_name_prefix(f): f for f in drunk_folders}
    
    # Find common prefixes
    common_prefixes = set(sober_dict.keys()) & set(drunk_dict.keys())
    
    for prefix in common_prefixes:
        matches[prefix] = {
            'sober': sober_dict[prefix],
            'drunk': drunk_dict[prefix]
        }
    
    return matches

def load_thermal_frames(person_path):
    frames = []
    frame_files = sorted([f for f in os.listdir(person_path) if f.endswith('.png')])
    
    for frame_file in frame_files:
        frame_path = os.path.join(person_path, frame_file)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            frames.append(frame)
    
    return np.array(frames) if frames else None

def extract_face_roi(frame, top=0.1, bottom=0.7, left=0.15, right=0.85):
    h, w = frame.shape
    y1 = int(h * top)
    y2 = int(h * bottom)
    x1 = int(w * left)
    x2 = int(w * right)
    return frame[y1:y2, x1:x2]

def extract_eye_roi(frame, side='left'):
    h, w = frame.shape
    if side == 'left':
        x1, x2 = int(w * 0.15), int(w * 0.4)
    else:
        x1, x2 = int(w * 0.6), int(w * 0.85)
    y1, y2 = int(h * 0.25), int(h * 0.4)
    return frame[y1:y2, x1:x2]

def calculate_temperature_features(frames):
    if frames is None or len(frames) == 0:
        return None
    
    features = {}
    
    # Convert to float for better statistics
    frames_float = frames.astype(np.float32)
    
    # Full frame statistics
    features['mean_temp'] = np.mean(frames_float)
    features['std_temp'] = np.std(frames_float)
    features['max_temp'] = np.max(frames_float)
    features['min_temp'] = np.min(frames_float)
    
    # Face ROI
    face_frames = np.array([extract_face_roi(f) for f in frames])
    features['face_mean'] = np.mean(face_frames)
    features['face_std'] = np.std(face_frames)
    
    # Left eye ROI
    left_eye_frames = np.array([extract_eye_roi(f, side='left') for f in frames])
    features['left_eye_mean'] = np.mean(left_eye_frames)
    features['left_eye_std'] = np.std(left_eye_frames)
    
    # Right eye ROI
    right_eye_frames = np.array([extract_eye_roi(f, side='right') for f in frames])
    features['right_eye_mean'] = np.mean(right_eye_frames)
    features['right_eye_std'] = np.std(right_eye_frames)
    
    # Average eye temp
    features['eye_avg_mean'] = (features['left_eye_mean'] + features['right_eye_mean']) / 2
    
    # Temperature gradient (change over frames)
    frame_means = np.mean(frames_float, axis=(1, 2))
    features['temp_trend'] = np.polyfit(range(len(frame_means)), frame_means, 1)[0]
    
    return features

def compare_conditions(sober_features, drunk_features):
    comparison = {}
    
    for key in sober_features:
        sober_val = sober_features[key]
        drunk_val = drunk_features[key]
        
        if sober_val != 0:
            pct_change = ((drunk_val - sober_val) / abs(sober_val)) * 100
        else:
            pct_change = 0
        
        comparison[key] = {
            'sober': sober_val,
            'drunk': drunk_val,
            'difference': drunk_val - sober_val,
            'pct_change': pct_change
        }
    
    return comparison

def visualize_comparison(person_name, sober_frames, drunk_frames, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Thermal Imaging: Sober vs Post-Alcohol\nSubject: {person_name}', 
                 fontsize=14, fontweight='bold')
    
    # Sample frames (first, middle, last)
    indices = [0, len(sober_frames)//2, len(sober_frames)-1]
    labels = ['Start', 'Middle', 'End']
    
    for idx, frame_idx in enumerate(indices):
        # Sober
        ax = axes[0, idx]
        im1 = ax.imshow(sober_frames[frame_idx], cmap='hot')
        ax.set_title(f'Sober - {labels[idx]}')
        ax.axis('off')
        plt.colorbar(im1, ax=ax, label='Intensity')
        
        # Drunk
        ax = axes[1, idx]
        im2 = ax.imshow(drunk_frames[frame_idx], cmap='hot')
        ax.set_title(f'Post-Alcohol - {labels[idx]}')
        ax.axis('off')
        plt.colorbar(im2, ax=ax, label='Intensity')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{person_name}_thermal_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    print("THERMAL IMAGING ANALYSIS - DRUNKENNESS DETECTION")    
    # Get person folders
    sober_persons = get_person_folders(SOBER_DIR)
    drunk_persons = get_person_folders(DRUNK_DIR)
    
    # Match by name prefix
    matched_pairs = match_folders(sober_persons, drunk_persons)
    
    print(f"\nFound {len(matched_pairs)} subjects with both sober and post-alcohol data")
    print(f"Sober baseline: {len(sober_persons)} subjects")
    print(f"Post-alcohol (4 glasses): {len(drunk_persons)} subjects")
    
    if len(matched_pairs) == 0:
        print("\nERROR: No matching folders found!")
        return
    
    # Store all comparisons
    all_comparisons = []
    
    print("PROCESSING SUBJECTS...")
    
    for idx, (prefix, folders) in enumerate(sorted(matched_pairs.items()), 1):
        person_name = prefix.split('_')[0]  # Get just the name part
        
        print(f"\n[{idx}] Processing: {person_name}")
        print(f"Sober folder:  {folders['sober']}")
        print(f"Drunk folder:  {folders['drunk']}")
        
        # Load frames
        sober_path = os.path.join(SOBER_DIR, folders['sober'])
        drunk_path = os.path.join(DRUNK_DIR, folders['drunk'])
        
        sober_frames = load_thermal_frames(sober_path)
        drunk_frames = load_thermal_frames(drunk_path)
        
        if sober_frames is None or drunk_frames is None:
            print(f"Failed to load frames")
            continue
        
        print(f"Loaded {len(sober_frames)} sober frames")
        print(f"Loaded {len(drunk_frames)} post-alcohol frames")
        
        # Extract features
        sober_features = calculate_temperature_features(sober_frames)
        drunk_features = calculate_temperature_features(drunk_frames)
        
        if sober_features is None or drunk_features is None:
            print(f"Failed to extract features")
            continue
        
        # Compare
        comparison = compare_conditions(sober_features, drunk_features)
        all_comparisons.append(comparison)
        
        # Visualizations
        viz_file = visualize_comparison(person_name, sober_frames, drunk_frames, OUTPUT_DIR)
        print(f"Saved visualization: {os.path.basename(viz_file)}")
        
        # Print summary
        print(f"Mean Temp Change: {comparison['mean_temp']['pct_change']:+.2f}%")
        print(f"Face Temp Change: {comparison['face_mean']['pct_change']:+.2f}%")
        print(f"Eye Temp Change: {comparison['eye_avg_mean']['pct_change']:+.2f}%")

    print("AGGREGATE STATISTICS")

    if all_comparisons:
        # Calculate mean changes across subjects
        mean_temp_changes = [c['mean_temp']['pct_change'] for c in all_comparisons]
        face_temp_changes = [c['face_mean']['pct_change'] for c in all_comparisons]
        eye_temp_changes = [c['eye_avg_mean']['pct_change'] for c in all_comparisons]
        
        print(f"\nTemperature Changes (Sober to Post-Alcohol) across {len(all_comparisons)} subjects:")
        print(f"Full Frame:  {np.mean(mean_temp_changes):+.2f}%")
        print(f"Face Region: {np.mean(face_temp_changes):+.2f}%")
        print(f"Eye Region:  {np.mean(eye_temp_changes):+.2f}%")
    print(f"Analysis complete! Results saved to:\n   {OUTPUT_DIR}")

if __name__ == "__main__":
    main()