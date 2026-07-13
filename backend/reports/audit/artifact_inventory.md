# Artifact Inventory & Integrity Report

Documents the physical existence and integrity of checkpoints, graphs, and metric files.

## Artifact Checklist

| Filename | Description | Status | File Size |
|----------|-------------|--------|-----------|
| `anomaly_model.pth` | Model Checkpoint: LSTM Autoencoder | FOUND | 144,993 bytes |
| `threshold.npy` | Autoencoder Anomaly Threshold Data | FOUND | 132 bytes |
| `model_type.txt` | Saved Anomaly Model Type | FOUND | 16 bytes |
| `classifier.pkl` | Model Checkpoint: XGBoost Classifier | FOUND | 649,211 bytes |
| `classifier_type.txt` | Saved Classifier Model Type | FOUND | 7 bytes |
| `ppo_response_agent.zip` | Model Checkpoint: PPO Response Agent | FOUND | 146,667 bytes |
| `anomaly_roc.png` | ROC Curve for Anomaly Detection | FOUND | 32,264 bytes |
| `anomaly_error_hist.png` | Reconstruction Error Distribution Histogram | FOUND | 18,718 bytes |
| `classifier_comparison.png` | F1 vs Latency Comparison Chart | FOUND | 33,563 bytes |
| `classifier_confusion.png` | Confusion Matrix for Classifier | FOUND | 42,061 bytes |
| `summary.json` | Summary Metrics File | FOUND | 473 bytes |

## Verification Summary
All core models, threshold files, and verification graphs are correctly present and non-empty.
