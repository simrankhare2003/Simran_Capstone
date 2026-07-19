import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

CSV_PATH = r"D:\rPPG_Results\rppg_features.csv"
OUTPUT_DIR = r"D:\Classification_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features to use
FEATURES = ['heart_rate', 'hrv_sdnn', 'hrv_rmssd', 'signal_mean', 'signal_std', 'signal_min', 'signal_max']

# Random seed for reproducibility
RANDOM_STATE = 42

df = pd.read_csv(CSV_PATH)

print(f"Total videos: {len(df)}")
print(f"Drunk: {len(df[df['category'] == 'drunk'])}")
print(f"Sober: {len(df[df['category'] == 'sober'])}\n")

# Extract features and labels
X = df[FEATURES].values
y = (df['category'] == 'drunk').astype(int).values  # 1 for drunk, 0 for sober

print(f"Features shape: {X.shape}")
print(f"Features used: {FEATURES}\n")

# Check for missing values
if np.any(np.isnan(X)):
    print("Found NaN values in features")
    # Replace NaN with mean
    nan_mask = np.isnan(X)
    col_mean = np.nanmean(X, axis=0)
    X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    print("Replaced NaN with column mean\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"Train set: {len(X_train)} videos ({len(X_train[y_train==1])} drunk, {len(X_train[y_train==0])} sober)")
print(f"Test set: {len(X_test)} videos ({len(X_test[y_test==1])} drunk, {len(X_test[y_test==0])} sober)\n")






drunk_data = df[df['category'] == 'drunk'][FEATURES]
sober_data = df[df['category'] == 'sober'][FEATURES]
 
# Calculate means
drunk_mean = drunk_data.mean()
sober_mean = sober_data.mean()
 
print("MEAN VALUES:\n")
print("Drunk Group:")
for feature, value in drunk_mean.items():
    print(f"  {feature:15s}: {value:8.4f}")
 
print("\nSober Group:")
for feature, value in sober_mean.items():
    print(f"  {feature:15s}: {value:8.4f}")
 
# Calculate covariance matrices
drunk_cov = drunk_data.cov()
sober_cov = sober_data.cov()
 
print("COVARIANCE MATRICES")

print("Drunk Group Covariance Matrix:")
print(drunk_cov.round(4))
 
print("\n\nSober Group Covariance Matrix:")
print(sober_cov.round(4))





# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)
svm_model.fit(X_train_scaled, y_train)

# Evaluate SVM
y_pred_svm = svm_model.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf_model.fit(X_train, y_train)  

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print("MODEL COMPARISON")

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'SVM': [svm_accuracy, svm_precision, svm_recall, svm_f1],
    'Random Forest': [rf_accuracy, rf_precision, rf_recall, rf_f1]
})

print(comparison_df.to_string(index=False))
print()

# Determine best model
if rf_f1 > svm_f1:
    print(f"Best Model: Random Forest (F1-Score: {rf_f1:.4f})")
else:
    print(f"Best Model: SVM (F1-Score: {svm_f1:.4f})")

print()

# Save metrics to CSV
results_csv = os.path.join(OUTPUT_DIR, "classification_results.csv")
comparison_df.to_csv(results_csv, index=False)

# Save detailed report
report_file = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_file, 'w') as f:

    f.write("DRUNK/SOBER CLASSIFICATION REPORT\n")
    
    f.write("DATASET:\n")
    f.write(f"Total videos: {len(df)}\n")
    f.write(f"Drunk: {len(df[df['category'] == 'drunk'])}\n")
    f.write(f"Sober: {len(df[df['category'] == 'sober'])}\n\n")
    
    f.write("TRAIN/TEST SPLIT (70-30):\n")
    f.write(f"Train: {len(X_train)} videos\n")
    f.write(f"Test: {len(X_test)} videos\n\n")
    
    f.write("FEATURES USED:\n")
    for feat in FEATURES:
        f.write(f"  - {feat}\n")
    f.write("\n")
    
    f.write("SVM RESULTS:\n")
    f.write(f"  Accuracy:  {svm_accuracy:.4f}\n")
    f.write(f"  Precision: {svm_precision:.4f}\n")
    f.write(f"  Recall:    {svm_recall:.4f}\n")
    f.write(f"  F1-Score:  {svm_f1:.4f}\n\n")
    
    f.write("RANDOM FOREST RESULTS:\n")
    f.write(f"  Accuracy:  {rf_accuracy:.4f}\n")
    f.write(f"  Precision: {rf_precision:.4f}\n")
    f.write(f"  Recall:    {rf_recall:.4f}\n")
    f.write(f"  F1-Score:  {rf_f1:.4f}\n\n")
