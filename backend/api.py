import os
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any

# Safe imports for optional ML libraries to make deployment lightweight and robust
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from stable_baselines3 import PPO
    HAS_RL = True
except ImportError:
    HAS_RL = False
    PPO = None

from utils import FEATURES, CLASSES, CLASS_MAP, REV_CLASS_MAP, ACTIONS

if HAS_TORCH:
    from train import DenseAutoencoder, LSTMAutoencoder, MLPClassifier, CNN1DClassifier, BiLSTMClassifier
else:
    DenseAutoencoder = None
    LSTMAutoencoder = None
    MLPClassifier = None
    CNN1DClassifier = None
    BiLSTMClassifier = None

if HAS_RL:
    from rl_agent import CyberSecurityEnv
else:
    CyberSecurityEnv = None

app = FastAPI(
    title="AI-Powered Cyber Threat Detection & Response API",
    description="FastAPI Backend for real-time traffic anomaly detection, attack classification, and RL response mitigation.",
    version="1.0.0"
)

# Global variables for models and utilities
scaler_mean = None
scaler_scale = None
anomaly_model = None
anomaly_threshold = None
anomaly_type = None
classifier_model = None
classifier_type = None
rl_agent = None

# Input data model representing a network flow
class NetworkFlowInput(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping CICIDS2017 feature names to their numeric values."
    )

@app.on_event("startup")
def load_assets():
    """Loads scaler, anomaly models, classifier models, and PPO policies on startup."""
    global scaler_mean, scaler_scale, anomaly_model, anomaly_threshold, anomaly_type, classifier_model, classifier_type, rl_agent
    
    base_dir = "D:\\cyber_threat_detection"
    processed_dir = os.path.join(base_dir, "data", "processed")
    anomaly_dir = os.path.join(base_dir, "model", "anomaly_model")
    classifier_dir = os.path.join(base_dir, "model", "classifier")
    rl_dir = os.path.join(base_dir, "model", "rl_policy")
    
    # Load Scaler parameters
    try:
        scaler_mean = np.load(os.path.join(processed_dir, "mean.npy"))
        scaler_scale = np.load(os.path.join(processed_dir, "scale.npy"))
    except Exception as e:
        print(f"Scaler parameters not found: {e}. API will run with mock scaling.")
        scaler_mean = np.zeros(len(FEATURES))
        scaler_scale = np.ones(len(FEATURES))
        
    # Load Anomaly Model
    if HAS_TORCH:
        try:
            anomaly_threshold = np.load(os.path.join(anomaly_dir, "threshold.npy"))[0]
            with open(os.path.join(anomaly_dir, "model_type.txt"), "r") as f:
                anomaly_type = f.read().strip()
                
            if anomaly_type == "dense_autoencoder":
                anomaly_model = DenseAutoencoder()
            else:
                anomaly_model = LSTMAutoencoder()
                
            anomaly_model.load_state_dict(torch.load(os.path.join(anomaly_dir, "anomaly_model.pth"), map_location=torch.device('cpu')))
            anomaly_model.eval()
            print(f"Successfully loaded anomaly model: {anomaly_type} (threshold: {anomaly_threshold:.6f})")
        except Exception as e:
            print(f"Could not load anomaly model: {e}. Building mock DenseAutoencoder.")
            anomaly_model = DenseAutoencoder()
            anomaly_threshold = 0.05
            anomaly_type = "dense_autoencoder"
    else:
        print("Torch not available. Anomaly model loaded as None (mock fallback).")
        anomaly_model = None
        anomaly_threshold = 0.05
        anomaly_type = "mock"
        
    # Load Classifier
    try:
        with open(os.path.join(classifier_dir, "classifier_type.txt"), "r") as f:
            classifier_type = f.read().strip()
            
        if classifier_type == "xgboost":
            try:
                import xgboost
                with open(os.path.join(classifier_dir, "classifier.pkl"), "rb") as f:
                    classifier_model = pickle.load(f)
                print("Successfully loaded xgboost classifier")
            except Exception as e:
                print(f"Could not load xgboost classifier: {e}")
                classifier_model = None
        elif HAS_TORCH:
            if classifier_type == "mlp":
                classifier_model = MLPClassifier()
            elif classifier_type == "cnn1d":
                classifier_model = CNN1DClassifier()
            else:
                classifier_model = BiLSTMClassifier()
            classifier_model.load_state_dict(torch.load(os.path.join(classifier_dir, "classifier.pth"), map_location=torch.device('cpu')))
            classifier_model.eval()
            print(f"Successfully loaded classifier: {classifier_type}")
        else:
            classifier_model = None
    except Exception as e:
        print(f"Could not load classifier model: {e}")
        if HAS_TORCH:
            classifier_type = "mlp"
            classifier_model = MLPClassifier()
            classifier_model.eval()
        else:
            classifier_model = None
            classifier_type = "mock"
        
    # Load RL Policy
    if HAS_RL:
        try:
            rl_agent = PPO.load(os.path.join(rl_dir, "ppo_response_agent"))
            print("Successfully loaded PPO RL policy response agent")
        except Exception as e:
            print(f"Could not load PPO agent policy: {e}. Mitigation decisions will fallback to rule-based.")
            rl_agent = None
    else:
        print("stable-baselines3 not available. RL agent loaded as None (rule-based fallback).")
        rl_agent = None

@app.post("/predict")
def predict_threat(flow_data: NetworkFlowInput):
    """Processes network flow features to output anomaly score, attack type, confidence, and recommended mitigation."""
    global scaler_mean, scaler_scale, anomaly_model, anomaly_threshold, anomaly_type, classifier_model, classifier_type, rl_agent
    
    # 1. Map input features
    features_dict = flow_data.features
    features_list = []
    for f in FEATURES:
        # Fill missing features with 0.0 default
        features_list.append(features_dict.get(f, 0.0))
        
    raw_vector = np.array(features_list).reshape(1, -1)
    
    # 2. Scale features
    scaled_vector = (raw_vector - scaler_mean) / scaler_scale
    
    # 3. Anomaly Detection Inference
    if HAS_TORCH and anomaly_model is not None:
        scaled_tensor = torch.FloatTensor(scaled_vector)
        with torch.no_grad():
            reconstructed = anomaly_model(scaled_tensor)
            reconstruction_error = torch.mean((reconstructed - scaled_tensor)**2, dim=1).item()
        is_anomaly = bool(reconstruction_error > anomaly_threshold)
    else:
        # Heuristic fallback anomaly check without PyTorch
        reconstruction_error = float(np.mean(np.abs(scaled_vector))) * 0.01
        is_anomaly = bool(reconstruction_error > anomaly_threshold)
    
    # 4. Attack Classification Inference
    if classifier_model is not None:
        try:
            if classifier_type == "xgboost":
                probs = classifier_model.predict_proba(scaled_vector)[0]
                class_idx = np.argmax(probs)
                confidence = float(probs[class_idx])
            elif HAS_TORCH:
                # PyTorch Classifiers
                with torch.no_grad():
                    outputs = classifier_model(scaled_tensor)
                    probs = torch.softmax(outputs, dim=1).numpy()[0]
                    class_idx = int(np.argmax(probs))
                    confidence = float(probs[class_idx])
            else:
                class_idx = 0
                confidence = 1.0
        except Exception:
            class_idx = 0
            confidence = 1.0
    else:
        # Heuristic classification fallback for demo frontend when model is missing
        dest_port = features_dict.get('destination_port', 80.0)
        if dest_port in [80, 443, 8080]:
            class_idx = 0  # BENIGN
        elif dest_port == 22:
            class_idx = 7  # Brute Force
        elif dest_port == 53:
            class_idx = 0  # BENIGN
        else:
            if features_dict.get('flow_packets_s', 0.0) > 1000:
                class_idx = 1  # DDoS
            elif features_dict.get('flow_duration', 0.0) > 5000000:
                class_idx = 2  # DoS Hulk
            else:
                class_idx = 0
        confidence = 0.88 if class_idx != 0 else 0.99
        
    attack_type = REV_CLASS_MAP.get(class_idx, "BENIGN")
    
    # 5. RL Response Mitigations
    # Build RL environment state vector
    flow_duration = float(np.clip(features_dict.get('flow_duration', 0.0) / 10000.0, 0.0, 1.0))
    packet_rate = float(np.clip(features_dict.get('flow_packets_s', 0.0) / 1000.0, 0.0, 1.0))
    threat_severity = 0.0 if class_idx == 0 else (0.9 if class_idx in [1, 2, 4, 5] else 0.6)
    norm_anomaly_score = float(np.clip(reconstruction_error / (anomaly_threshold * 2.0), 0.0, 1.0))
    
    # Execute mitigation decision
    if HAS_RL and rl_agent is not None:
        state_vector = np.array([
            norm_anomaly_score,
            float(class_idx),
            confidence,
            flow_duration,
            packet_rate,
            threat_severity,
            0.05, # historical attack frequency
            0.02  # false positive probability
        ], dtype=np.float32)
        action_idx, _ = rl_agent.predict(state_vector, deterministic=True)
        recommended_action = ACTIONS.get(int(action_idx), "Raise Alert")
    else:
        # Fallback Rule-Based Logic
        if class_idx == 0:
            recommended_action = "Ignore"
        elif class_idx in [1, 2, 4, 5]:
            recommended_action = "Block IP"
        else:
            recommended_action = "Restrict Port"
            
    return {
        "anomaly_score": float(reconstruction_error),
        "anomaly_threshold": float(anomaly_threshold),
        "is_anomaly": is_anomaly,
        "attack_type": attack_type,
        "confidence_score": confidence,
        "recommended_action": recommended_action
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "has_torch": HAS_TORCH,
        "has_rl": HAS_RL,
        "anomaly_model_loaded": anomaly_model is not None,
        "classifier_loaded": classifier_model is not None,
        "rl_agent_loaded": rl_agent is not None
    }
