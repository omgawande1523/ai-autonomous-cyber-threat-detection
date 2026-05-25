import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def run_experiment(base_dir="D:\\cyber_threat_detection"):
    print("Running model baseline experiments...")
    
    processed_dir = os.path.join(base_dir, "data", "processed")
    X_train_path = os.path.join(processed_dir, "X_train.csv")
    
    if not os.path.exists(X_train_path):
        print("Processed data not found. Running pre-processing pipeline first...")
        from utils import download_and_preprocess_dataset
        download_and_preprocess_dataset(base_dir)
        
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv")).values
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv")).values
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.flatten()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.flatten()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Train a baseline XGBoost classifier on a sample to make it fast
    print("Training Baseline XGBoost Classifier on subset...")
    sample_idx = np.random.choice(len(X_train), size=min(len(X_train), 2000), replace=False)
    X_train_sub = X_train[sample_idx]
    y_train_sub = y_train[sample_idx]
    
    clf = XGBClassifier(max_depth=3, n_estimators=50, random_state=42)
    clf.fit(X_train_sub, y_train_sub)
    
    preds = clf.predict(X_test)
    
    print("\nBaseline Experiment Results (XGBoost):")
    from utils import CLASSES
    # Filter classes that are in y_test
    unique_labels = np.unique(np.concatenate([y_test, preds]))
    target_names = [CLASSES[i] for i in unique_labels]
    
    report = classification_report(y_test, preds, labels=unique_labels, target_names=target_names)
    print(report)

if __name__ == "__main__":
    run_experiment()
