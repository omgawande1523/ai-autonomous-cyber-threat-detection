import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

from utils import download_and_preprocess_dataset, FEATURES, CLASSES, CLASS_MAP, REV_CLASS_MAP

# Ensure CUDA or CPU is selected
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# CPU-SAFE CONFIGURATIONS & HYPERPARAMETERS
# ==========================================
MAX_EPOCHS_AE = 10
MAX_EPOCHS_CLF = 10
AE_PATIENCE = 2
CLF_PATIENCE = 2

# Batch size auto-adjustment based on RAM
def get_safe_batch_size():
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        print(f"System memory available: {available_gb:.2f} GB")
        if available_gb < 2.0:
            return 32
        elif available_gb < 4.0:
            return 64
        else:
            return 128
    except Exception:
        print("psutil not available, defaulting to conservative batch size: 64")
        return 64

BATCH_SIZE = get_safe_batch_size()
print(f"Configured Batch Size: {BATCH_SIZE}")

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=2, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

# ==========================================
# 1. ANOMALY DETECTION MODEL DEFINITIONS
# ==========================================

class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim=len(FEATURES)):
        super(DenseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=len(FEATURES)):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, 16, batch_first=True)
        self.decoder_lstm = nn.LSTM(16, input_dim, batch_first=True)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder_lstm(encoded)
        return decoded.squeeze(1)

# ==========================================
# 2. CLASSIFICATION MODEL DEFINITIONS
# ==========================================

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=len(FEATURES), num_classes=len(CLASSES)):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim=len(FEATURES), num_classes=len(CLASSES)):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        conv_out_len = input_dim // 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * conv_out_len, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.fc(x)
        return x

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=len(FEATURES), num_classes=len(CLASSES)):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, 16, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out.squeeze(1)
        return self.fc(out)


# ==========================================
# 3. TRAINING & EVALUATION FUNCTIONS
# ==========================================

def train_anomaly_model(model_type, X_train_benign, X_test, y_test, writer, base_dir="D:\\cyber_threat_detection"):
    print(f"\n--- Training Anomaly Detector: {model_type} ---")
    
    # Split training into train and validation for early stopping
    X_tr, X_val = train_test_split(X_train_benign, test_size=0.1, random_state=42)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_tr))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if model_type == "dense_autoencoder":
        model = DenseAutoencoder().to(device)
    else:
        model = LSTMAutoencoder().to(device)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    early_stopping = EarlyStopping(patience=AE_PATIENCE)
    
    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(base_dir, "model", "anomaly_model", f"temp_{model_type}.pth")
    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    
    for epoch in range(MAX_EPOCHS_AE):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{MAX_EPOCHS_AE} - Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        writer.add_scalar(f"Anomaly_{model_type}/TrainLoss", avg_train, epoch)
        writer.add_scalar(f"Anomaly_{model_type}/ValLoss", avg_val, epoch)
        
        # Checkpoint if validation loss improves
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_checkpoint_path)
            
        # Early Stopping check
        if early_stopping(avg_val):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    # Load best checkpoint
    model.load_state_dict(torch.load(best_checkpoint_path))
    model.eval()
    
    # Calculate threshold (95th percentile of validation reconstruction error)
    with torch.no_grad():
        val_inputs = torch.FloatTensor(X_val).to(device)
        val_reconstructions = model(val_inputs)
        val_errors = torch.mean((val_reconstructions - val_inputs)**2, dim=1).cpu().numpy()
        threshold = np.percentile(val_errors, 95)
        
        # Evaluate on full test set
        test_inputs = torch.FloatTensor(X_test).to(device)
        test_reconstructions = model(test_inputs)
        test_errors = torch.mean((test_reconstructions - test_inputs)**2, dim=1).cpu().numpy()
        
    y_test_anomaly = (y_test != CLASS_MAP['BENIGN']).astype(int)
    predictions = (test_errors > threshold).astype(int)
    
    acc = accuracy_score(y_test_anomaly, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_anomaly, predictions, average='binary', zero_division=0)
    auc = roc_auc_score(y_test_anomaly, test_errors)
    
    print(f"{model_type} evaluation: Threshold = {threshold:.6f} | F1 = {f1:.4f} | AUC = {auc:.4f}")
    
    # Clean up temp file
    if os.path.exists(best_checkpoint_path):
        try:
            os.remove(best_checkpoint_path)
        except Exception:
            pass
            
    return model, threshold, f1, auc, test_errors, predictions, y_test_anomaly


def train_classifier(model_type, X_train, y_train, X_test, y_test, writer, base_dir="D:\\cyber_threat_detection"):
    print(f"\n--- Training Classifier: {model_type} ---")
    
    if model_type == "xgboost":
        from sklearn.utils.class_weight import compute_class_weight
        classes_present = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y_train)
        weight_dict = {c: w for c, w in zip(classes_present, weights)}
        sample_weight = np.array([weight_dict[y] for y in y_train])
        
        # CPU-safe fast XGBoost
        model = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1, max_depth=3, n_estimators=100)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        
        inf_start = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - inf_start) / len(X_test) * 1000 # ms/sample
        
        y_pred_proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        print(f"XGBoost F1: {f1:.4f} | Latency: {inference_time:.4f} ms")
        writer.add_scalar("Classifier_xgboost/Accuracy", acc, 0)
        writer.add_scalar("Classifier_xgboost/F1", f1, 0)
        return model, f1, auc, inference_time, y_pred
        
    # PyTorch DL Classifiers
    # Split training into train and validation
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(CLASSES) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    if model_type == "mlp":
        model = MLPClassifier().to(device)
    elif model_type == "cnn1d":
        model = CNN1DClassifier().to(device)
    else:
        model = BiLSTMClassifier().to(device)
        
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    early_stopping = EarlyStopping(patience=CLF_PATIENCE)
    
    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(base_dir, "model", "classifier", f"temp_{model_type}.pth")
    os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
    
    for epoch in range(MAX_EPOCHS_CLF):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{MAX_EPOCHS_CLF} - Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        writer.add_scalar(f"Classifier_{model_type}/TrainLoss", avg_train, epoch)
        writer.add_scalar(f"Classifier_{model_type}/ValLoss", avg_val, epoch)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_checkpoint_path)
            
        if early_stopping(avg_val):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    # Load best checkpoint
    model.load_state_dict(torch.load(best_checkpoint_path))
    model.eval()
    
    y_pred = []
    y_probs = []
    inference_times = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            sample_x = torch.FloatTensor(X_test[i]).unsqueeze(0).to(device)
            start_inf = time.time()
            output = model(sample_x)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred = np.argmax(prob)
            inf_time = (time.time() - start_inf) * 1000 # ms
            
            y_pred.append(pred)
            y_probs.append(prob)
            inference_times.append(inf_time)
            
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    inference_time = np.mean(inference_times)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
    
    print(f"{model_type} evaluation: F1 = {f1:.4f} | Latency = {inference_time:.4f} ms")
    writer.add_scalar(f"Classifier_{model_type}/Accuracy", acc, MAX_EPOCHS_CLF)
    writer.add_scalar(f"Classifier_{model_type}/F1", f1, MAX_EPOCHS_CLF)
    
    # Clean up temp file
    if os.path.exists(best_checkpoint_path):
        try:
            os.remove(best_checkpoint_path)
        except Exception:
            pass
            
    return model, f1, auc, inference_time, y_pred


# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================

def run_pipeline(base_dir="D:\\cyber_threat_detection"):
    log_dir = os.path.join(base_dir, "logs")
    writer = SummaryWriter(log_dir)
    
    # Step 1: Preprocess dataset
    num_samples, train_shape, test_shape = download_and_preprocess_dataset(base_dir)
    
    # Load processed data
    processed_dir = os.path.join(base_dir, "data", "processed")
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv")).values
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv")).values
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.flatten()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.flatten()
    X_train_benign = pd.read_csv(os.path.join(processed_dir, "X_train_benign.csv")).values
    
    # Create directories
    model_anomaly_dir = os.path.join(base_dir, "model", "anomaly_model")
    model_classifier_dir = os.path.join(base_dir, "model", "classifier")
    reports_graphs_dir = os.path.join(base_dir, "reports", "graphs")
    reports_metrics_dir = os.path.join(base_dir, "reports", "metrics")
    
    os.makedirs(model_anomaly_dir, exist_ok=True)
    os.makedirs(model_classifier_dir, exist_ok=True)
    os.makedirs(reports_graphs_dir, exist_ok=True)
    os.makedirs(reports_metrics_dir, exist_ok=True)
    
    # Train Anomaly Detection Models
    dense_ae, dense_th, dense_f1, dense_auc, dense_errs, dense_pred, y_test_anom = train_anomaly_model(
        "dense_autoencoder", X_train_benign, X_test, y_test, writer, base_dir
    )
    lstm_ae, lstm_th, lstm_f1, lstm_auc, lstm_errs, lstm_pred, _ = train_anomaly_model(
        "lstm_autoencoder", X_train_benign, X_test, y_test, writer, base_dir
    )
    
    # Select best anomaly model
    best_anomaly_model = dense_ae
    best_anomaly_name = "dense_autoencoder"
    best_anomaly_th = dense_th
    best_anomaly_f1 = dense_f1
    best_anomaly_errs = dense_errs
    best_anomaly_pred = dense_pred
    
    if lstm_f1 > dense_f1:
        best_anomaly_model = lstm_ae
        best_anomaly_name = "lstm_autoencoder"
        best_anomaly_th = lstm_th
        best_anomaly_f1 = lstm_f1
        best_anomaly_errs = lstm_errs
        best_anomaly_pred = lstm_pred
        
    print(f"\n>>> Best Anomaly Model Selected: {best_anomaly_name} (F1: {best_anomaly_f1:.4f})")
    
    # Save best anomaly model & threshold
    torch.save(best_anomaly_model.state_dict(), os.path.join(model_anomaly_dir, "anomaly_model.pth"))
    np.save(os.path.join(model_anomaly_dir, "threshold.npy"), np.array([best_anomaly_th]))
    with open(os.path.join(model_anomaly_dir, "model_type.txt"), "w") as f:
        f.write(best_anomaly_name)
        
    # Save anomaly detection graphs
    fpr, tpr, _ = roc_curve(y_test_anom, best_anomaly_errs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{best_anomaly_name} (AUC = {roc_auc_score(y_test_anom, best_anomaly_errs):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Anomaly Detection ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(reports_graphs_dir, "anomaly_roc.png"))
    plt.close()
    
    plt.figure()
    plt.hist(best_anomaly_errs[y_test_anom == 0], bins=50, alpha=0.5, label='Benign', color='blue')
    plt.hist(best_anomaly_errs[y_test_anom == 1], bins=50, alpha=0.5, label='Anomaly', color='red')
    plt.axvline(best_anomaly_th, color='green', linestyle='dashed', linewidth=2, label=f'Threshold ({best_anomaly_th:.4f})')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join(reports_graphs_dir, "anomaly_error_hist.png"))
    plt.close()
    
    cm = confusion_matrix(y_test_anom, best_anomaly_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Anomaly'], yticklabels=['Benign', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Anomaly Confusion Matrix')
    plt.savefig(os.path.join(reports_graphs_dir, "anomaly_confusion.png"))
    plt.close()
    
    # Train Attack Classifiers
    mlp_model, mlp_f1, mlp_auc, mlp_lat, mlp_pred = train_classifier("mlp", X_train, y_train, X_test, y_test, writer, base_dir)
    cnn_model, cnn_f1, cnn_auc, cnn_lat, cnn_pred = train_classifier("cnn1d", X_train, y_train, X_test, y_test, writer, base_dir)
    lstm_model, lstm_f1, lstm_auc, lstm_lat, lstm_pred = train_classifier("bilstm", X_train, y_train, X_test, y_test, writer, base_dir)
    xgb_model, xgb_f1, xgb_auc, xgb_lat, xgb_pred = train_classifier("xgboost", X_train, y_train, X_test, y_test, writer, base_dir)
    
    # Select best classifier
    classifiers = {
        "mlp": (mlp_model, mlp_f1, mlp_auc, mlp_lat, mlp_pred),
        "cnn1d": (cnn_model, cnn_f1, cnn_auc, cnn_lat, cnn_pred),
        "bilstm": (lstm_model, lstm_f1, lstm_auc, lstm_lat, lstm_pred),
        "xgboost": (xgb_model, xgb_f1, xgb_auc, xgb_lat, xgb_pred)
    }
    
    best_classifier_name = max(classifiers, key=lambda k: classifiers[k][1])
    best_clf_model, best_clf_f1, best_clf_auc, best_clf_lat, best_clf_pred = classifiers[best_classifier_name]
    
    print(f"\n>>> Best Classifier Selected: {best_classifier_name} (Weighted F1: {best_clf_f1:.4f})")
    
    # Save best classifier
    if best_classifier_name == "xgboost":
        import pickle
        with open(os.path.join(model_classifier_dir, "classifier.pkl"), "wb") as f:
            pickle.dump(best_clf_model, f)
    else:
        torch.save(best_clf_model.state_dict(), os.path.join(model_classifier_dir, "classifier.pth"))
        
    with open(os.path.join(model_classifier_dir, "classifier_type.txt"), "w") as f:
        f.write(best_classifier_name)
        
    # Generate classifier comparison chart
    names = list(classifiers.keys())
    f1s = [classifiers[n][1] for n in names]
    lats = [classifiers[n][3] for n in names]
    
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('F1 Score', color=color)
    ax1.bar(names, f1s, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Inference Latency (ms)', color=color)
    ax2.plot(names, lats, color=color, marker='o', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Classifier Performance vs Latency Comparison')
    fig.tight_layout()
    plt.savefig(os.path.join(reports_graphs_dir, "classifier_comparison.png"))
    plt.close()
    
    # Generate Best Classifier Confusion Matrix
    cm_clf = confusion_matrix(y_test, best_clf_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_clf, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_classifier_name.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_graphs_dir, "classifier_confusion.png"))
    plt.close()
    
    # Save reports & metrics
    report = classification_report(y_test, best_clf_pred, target_names=CLASSES, zero_division=0)
    with open(os.path.join(reports_metrics_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        
    # Write summary metrics JSON
    import json
    metrics_summary = {
        "best_anomaly_detector": best_anomaly_name,
        "best_anomaly_f1": float(best_anomaly_f1),
        "best_anomaly_auc": float(roc_auc_score(y_test_anom, best_anomaly_errs)),
        "best_classifier": best_classifier_name,
        "best_classifier_f1": float(best_clf_f1),
        "best_classifier_auc": float(best_clf_auc),
        "best_classifier_latency_ms": float(best_clf_lat),
        "dataset_rows": num_samples,
        "train_shape": list(train_shape),
        "test_shape": list(test_shape)
    }
    with open(os.path.join(reports_metrics_dir, "summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    writer.close()
    print("Training pipeline run completed successfully!")

if __name__ == "__main__":
    run_pipeline()
