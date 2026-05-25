import os
import time
import json
import numpy as np
import pandas as pd
import torch
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Imports from project components
from utils import FEATURES, CLASSES, CLASS_MAP, REV_CLASS_MAP, ACTIONS, PacketSniffer, explain_with_captum, explain_with_shap
from train import DenseAutoencoder, LSTMAutoencoder, MLPClassifier, CNN1DClassifier, BiLSTMClassifier
from rl_agent import CyberSecurityEnv, load_rl_agent

# Set page config for premium widescreen dark layout
st.set_page_config(
    page_title="SOC Command Center | AI Cyber Threat Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium dark mode CSS with glassmorphism and cybersecurity accents
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Neon glow headings */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 5px rgba(88, 166, 255, 0.4);
    }
    
    /* Metrics panels */
    div[data-testid="stMetricValue"] {
        color: #58a6ff;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Custom Alert/Panel Cards */
    .soc-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .soc-card-header {
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-bottom: 12px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    .neon-border-green {
        border: 1px solid #238636;
        box-shadow: 0 0 8px rgba(35, 134, 54, 0.3);
    }
    .neon-border-red {
        border: 1px solid #f85149;
        box-shadow: 0 0 8px rgba(248, 81, 73, 0.3);
    }
    .neon-border-yellow {
        border: 1px solid #d29922;
        box-shadow: 0 0 8px rgba(210, 153, 34, 0.3);
    }
    
    /* Custom button styling */
    .stButton>button {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #58a6ff;
        color: #0d1117;
        box-shadow: 0 0 10px rgba(88, 166, 255, 0.6);
    }
</style>
""", unsafe_style_html=True)

# Cache model loading to optimize dashboard updates
@st.cache_resource
def load_models():
    base_dir = "D:\\cyber_threat_detection"
    processed_dir = os.path.join(base_dir, "data", "processed")
    anomaly_dir = os.path.join(base_dir, "model", "anomaly_model")
    classifier_dir = os.path.join(base_dir, "model", "classifier")
    
    # Load Scaler
    try:
        mean = np.load(os.path.join(processed_dir, "mean.npy"))
        scale = np.load(os.path.join(processed_dir, "scale.npy"))
    except:
        mean, scale = np.zeros(len(FEATURES)), np.ones(len(FEATURES))
        
    # Load Anomaly Threshold & Model
    try:
        threshold = np.load(os.path.join(anomaly_dir, "threshold.npy"))[0]
        with open(os.path.join(anomaly_dir, "model_type.txt"), "r") as f:
            anomaly_type = f.read().strip()
            
        if anomaly_type == "dense_autoencoder":
            anom_model = DenseAutoencoder()
        else:
            anom_model = LSTMAutoencoder()
        anom_model.load_state_dict(torch.load(os.path.join(anomaly_dir, "anomaly_model.pth"), map_location=torch.device('cpu')))
        anom_model.eval()
    except:
        anom_model = DenseAutoencoder()
        threshold = 0.05
        anomaly_type = "dense_autoencoder (Mock)"
        
    # Load Classifier
    try:
        with open(os.path.join(classifier_dir, "classifier_type.txt"), "r") as f:
            clf_type = f.read().strip()
            
        if clf_type == "xgboost":
            with open(os.path.join(classifier_dir, "classifier.pkl"), "rb") as f:
                clf_model = pickle.load(f)
        else:
            if clf_type == "mlp":
                clf_model = MLPClassifier()
            elif clf_type == "cnn1d":
                clf_model = CNN1DClassifier()
            else:
                clf_model = BiLSTMClassifier()
            clf_model.load_state_dict(torch.load(os.path.join(classifier_dir, "classifier.pth"), map_location=torch.device('cpu')))
            clf_model.eval()
    except:
        clf_model = MLPClassifier()
        clf_type = "mlp (Mock)"
        
    # Load RL Response Agent
    rl_agent = load_rl_agent(base_dir)
    
    return mean, scale, anom_model, threshold, anomaly_type, clf_model, clf_type, rl_agent

# Load models and scaling parameters
mean, scale, anom_model, threshold, anomaly_type, clf_model, clf_type, rl_agent = load_models()

# Setup Streamlit Session State variables
if 'processed_packets' not in st.session_state:
    st.session_state.processed_packets = 0
if 'detected_threats' not in st.session_state:
    st.session_state.detected_threats = 0
if 'threat_log' not in st.session_state:
    # Set list of dictionaries for packets log
    st.session_state.threat_log = []
if 'sniffing_active' not in st.session_state:
    st.session_state.sniffing_active = False
if 'sniffer' not in st.session_state:
    st.session_state.sniffer = None

# Custom packet handler callback
def packet_callback(features_array, true_class):
    # Scale inputs
    scaled = (features_array - mean) / scale
    
    # Run Anomaly Detection
    scaled_tensor = torch.FloatTensor(scaled)
    with torch.no_grad():
        recon = anom_model(scaled_tensor)
        recon_err = torch.mean((recon - scaled_tensor)**2, dim=1).item()
    is_anom = recon_err > threshold
    
    # Run Classification
    if clf_type == "xgboost":
        probs = clf_model.predict_proba(scaled)[0]
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
    else:
        with torch.no_grad():
            outputs = clf_model(scaled_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            class_idx = np.argmax(probs)
            confidence = probs[class_idx]
            
    pred_class = REV_CLASS_MAP.get(class_idx, "BENIGN")
    
    # Run RL Agent Mitigation Recommendation
    flow_duration = float(np.clip(features_array[0, 1] / 10000.0, 0.0, 1.0))
    packet_rate = float(np.clip(features_array[0, 15] / 1000.0, 0.0, 1.0))
    threat_severity = 0.0 if class_idx == 0 else (0.9 if class_idx in [1, 2, 4, 5] else 0.6)
    norm_anom = float(np.clip(recon_err / (threshold * 2.0), 0.0, 1.0))
    
    state_vector = np.array([
        norm_anom, float(class_idx), confidence, flow_duration, packet_rate, threat_severity, 0.05, 0.02
    ], dtype=np.float32)
    
    if rl_agent is not None:
        action_idx, _ = rl_agent.predict(state_vector, deterministic=True)
        mitigation = ACTIONS.get(int(action_idx), "Raise Alert")
    else:
        # Rule-based fallback
        mitigation = "Ignore" if class_idx == 0 else ("Block IP" if class_idx in [1, 2, 4, 5] else "Restrict Port")
        
    # Append log entry
    severity_level = "HIGH" if pred_class in ['DDoS', 'DoS Hulk', 'Bot', 'Infiltration'] else ("MEDIUM" if pred_class in ['PortScan', 'Brute Force', 'Web Attack'] else "LOW")
    
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "anomaly_score": recon_err,
        "is_anomaly": is_anom,
        "classification": pred_class,
        "confidence": confidence,
        "severity": severity_level,
        "mitigation": mitigation,
        "raw_features": features_array.tolist()[0]
    }
    
    st.session_state.threat_log.insert(0, log_entry)
    
    # Cap log at 100 entries
    if len(st.session_state.threat_log) > 100:
        st.session_state.threat_log.pop()
        
    st.session_state.processed_packets += 1
    if pred_class != 'BENIGN' or is_anom:
        st.session_state.detected_threats += 1

# Header banner
st.markdown("""
<div style="background-color: #161b22; border-bottom: 2px solid #58a6ff; padding: 15px; border-radius: 8px; margin-bottom: 25px;">
    <h1 style="margin: 0; display: inline-block;">🛡️ SOC Command Center</h1>
    <span style="float: right; font-family: monospace; color: #8b949e; padding-top: 10px;">AI-Powered Autonomous Threat Detection & Response</span>
</div>
""", unsafe_style_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### 🎛️ SYSTEM CONTROLS")

sniffer_mode = st.sidebar.radio("Sniffer Mode", ["Simulated Stream", "Live Network Sniffing"])
simulate_flag = (sniffer_mode == "Simulated Stream")

# Sniffer Control buttons
if st.sidebar.button("Start Threat Monitoring", disabled=st.session_state.sniffing_active):
    st.session_state.sniffing_active = True
    st.session_state.sniffer = PacketSniffer(
        callback=packet_callback,
        simulate=simulate_flag
    )
    st.session_state.sniffer.start()
    st.sidebar.success("Threat monitor started.")

if st.sidebar.button("Stop Threat Monitoring", disabled=not st.session_state.sniffing_active):
    st.session_state.sniffing_active = False
    if st.session_state.sniffer is not None:
        st.session_state.sniffer.stop()
        st.session_state.sniffer = None
    st.sidebar.warning("Threat monitor stopped.")

# System configuration details in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 ENGINE DETAILS")
st.sidebar.markdown(f"**Anomaly Detector**: `{anomaly_type}`")
st.sidebar.markdown(f"**Anomaly Threshold**: `{threshold:.5f}`")
st.sidebar.markdown(f"**Attack Classifier**: `{clf_type.upper()}`")
st.sidebar.markdown(f"**RL Mitigation Policy**: `PPO (Stable-Baselines3)`")

# ----------------- MAIN LAYOUT -----------------

# Dashboard tabs
tab_live, tab_sim, tab_model, tab_reports = st.tabs([
    "🟢 Live Ops Feed", 
    "🧪 Threat Simulator", 
    "🧠 Model Governance & XAI", 
    "📊 Reports & Audit Logs"
])

# ==================== TAB 1: LIVE OPS FEED ====================
with tab_live:
    # Dashboard summary row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Packets Analyzed", f"{st.session_state.processed_packets:,}")
    with col2:
        st.metric("Threats Intercepted", f"{st.session_state.detected_threats:,}")
    with col3:
        mean_err = np.mean([x['anomaly_score'] for x in st.session_state.threat_log]) if st.session_state.threat_log else 0.0
        st.metric("Mean Anomaly Score", f"{mean_err:.5f}")
    with col4:
        status_text = "ACTIVE SCANNING" if st.session_state.sniffing_active else "SCANNER OFFLINE"
        st.metric("System Defense State", status_text)

    st.markdown("---")

    # Real-time charts and lists grid
    chart_col, log_col = st.columns([7, 5])
    
    with chart_col:
        st.subheader("📊 Live Threat Activity & Severity Breakdown")
        
        # Threat history timeline chart
        if len(st.session_state.threat_log) > 0:
            df_log = pd.DataFrame(st.session_state.threat_log)
            # Reverse order for chronological display
            df_log_plot = df_log.iloc[::-1].copy()
            
            fig_timeline = px.line(
                df_log_plot, 
                x="timestamp", 
                y="anomaly_score",
                title="Real-Time Flow Anomaly Scores (Autoencoder Reconstruction Error)",
                color_discrete_sequence=["#58a6ff"]
            )
            fig_timeline.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            fig_timeline.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font_color="#c9d1d9",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Classification Distribution
            class_counts = df_log['classification'].value_counts().reset_index()
            class_counts.columns = ['Attack Class', 'Count']
            
            fig_dist = px.bar(
                class_counts,
                x='Attack Class',
                y='Count',
                title='Detected Attack Classes Summary',
                color='Count',
                color_continuous_scale=px.colors.sequential.Sunset
            )
            fig_dist.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font_color="#c9d1d9",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        else:
            st.info("Start threat monitoring to populate live dashboard visualization charts.")
            
    with log_col:
        st.subheader("🚨 Incident Mitigation Log Feed")
        
        if len(st.session_state.threat_log) > 0:
            for item in st.session_state.threat_log[:8]:
                border_class = "neon-border-red" if item['severity'] == "HIGH" else ("neon-border-yellow" if item['severity'] == "MEDIUM" else "neon-border-green")
                
                st.markdown(f"""
                <div class="soc-card {border_class}">
                    <div class="soc-card-header">[{item['timestamp']}] CLASSIFICATION: {item['classification'].upper()}</div>
                    <b>Severity:</b> {item['severity']} | <b>Anomaly Score:</b> {item['anomaly_score']:.5f}<br/>
                    <b>Confidence:</b> {item['confidence']*100:.1f}% | <b>Action Mitigated:</b> <code style="color:#58a6ff;">{item['mitigation']}</code>
                </div>
                """, unsafe_style_html=True)
        else:
            st.info("Waiting for network flows to be parsed...")

# ==================== TAB 2: THREAT SIMULATOR ====================
with tab_sim:
    st.subheader("🧪 Threat Vector Manual Entry & Simulator")
    st.markdown("Use this tab to input network flow statistics manually and inspect the complete prediction & mitigation chain.")
    
    # Input columns
    col_in1, col_in2, col_in3 = st.columns(3)
    
    input_vals = {}
    with col_in1:
        input_vals['destination_port'] = st.number_input("Destination Port", min_value=0, max_value=65535, value=80)
        input_vals['flow_duration'] = st.number_input("Flow Duration (µs)", min_value=0.0, value=1000.0)
        input_vals['total_fwd_packets'] = st.number_input("Total Forward Packets", min_value=0.0, value=5.0)
        input_vals['total_backward_packets'] = st.number_input("Total Backward Packets", min_value=0.0, value=5.0)
    with col_in2:
        input_vals['flow_packets_s'] = st.number_input("Flow Packets / s", min_value=0.0, value=20.0)
        input_vals['flow_bytes_s'] = st.number_input("Flow Bytes / s", min_value=0.0, value=1500.0)
        input_vals['syn_flag_count'] = st.selectbox("SYN Flag Count", [0.0, 1.0])
        input_vals['psh_flag_count'] = st.selectbox("PSH Flag Count", [0.0, 1.0])
    with col_in3:
        input_vals['fwd_packet_length_mean'] = st.number_input("Forward Packet Length Mean", min_value=0.0, value=64.0)
        input_vals['bwd_packet_length_mean'] = st.number_input("Backward Packet Length Mean", min_value=0.0, value=128.0)
        input_vals['down_up_ratio'] = st.number_input("Down/Up Ratio", min_value=0.0, value=1.0)
        
    if st.button("Simulate & Analyze Flow"):
        # Map values to full feature vector
        full_vector = {f: 0.0 for f in FEATURES}
        for k, v in input_vals.items():
            if k in full_vector:
                full_vector[k] = float(v)
                
        features_array = np.array([full_vector[f] for f in FEATURES]).reshape(1, -1)
        
        # Scaling
        scaled = (features_array - mean) / scale
        scaled_tensor = torch.FloatTensor(scaled)
        
        # Run predictions
        with torch.no_grad():
            recon = anom_model(scaled_tensor)
            recon_err = torch.mean((recon - scaled_tensor)**2, dim=1).item()
            is_anom = recon_err > threshold
            
            if clf_type == "xgboost":
                probs = clf_model.predict_proba(scaled)[0]
                class_idx = np.argmax(probs)
                confidence = probs[class_idx]
            else:
                outputs = clf_model(scaled_tensor)
                probs = torch.softmax(outputs, dim=1).numpy()[0]
                class_idx = np.argmax(probs)
                confidence = probs[class_idx]
                
        pred_class = REV_CLASS_MAP.get(class_idx, "BENIGN")
        
        # Query RL Response Agent
        flow_duration_val = float(np.clip(input_vals['flow_duration'] / 10000.0, 0.0, 1.0))
        packet_rate_val = float(np.clip(input_vals['flow_packets_s'] / 1000.0, 0.0, 1.0))
        threat_severity_val = 0.0 if class_idx == 0 else (0.9 if class_idx in [1, 2, 4, 5] else 0.6)
        norm_anom = float(np.clip(recon_err / (threshold * 2.0), 0.0, 1.0))
        
        state_vector = np.array([
            norm_anom, float(class_idx), confidence, flow_duration_val, packet_rate_val, threat_severity_val, 0.05, 0.02
        ], dtype=np.float32)
        
        if rl_agent is not None:
            action_idx, _ = rl_agent.predict(state_vector, deterministic=True)
            mitigation = ACTIONS.get(int(action_idx), "Raise Alert")
        else:
            mitigation = "Ignore" if class_idx == 0 else ("Block IP" if class_idx in [1, 2, 4, 5] else "Restrict Port")
            
        # Display simulated metrics
        st.markdown("---")
        st.markdown("### 🎛️ Analysis Results")
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            st.metric("Anomaly Score", f"{recon_err:.6f}")
            st.markdown(f"**Anomaly status:** `{'ANOMALY' if is_anom else 'NORMAL'}`")
        with col_res2:
            st.metric("Predicted Attack Type", pred_class)
            st.markdown(f"**Model Type:** `{clf_type}`")
        with col_res3:
            st.metric("Classification Confidence", f"{confidence*100:.2f}%")
        with col_res4:
            st.metric("Recommended Mitigation", mitigation)
            st.markdown("**Autonomous RL Agent decision**")
            
        # Explainable AI Segment
        st.markdown("---")
        st.markdown("### 🔍 Explainable AI (XAI) Attribution")
        
        # 1. Captum Integrated Gradients
        if clf_type != "xgboost":
            st.markdown("#### PyTorch Model Feature Attribution (Captum Integrated Gradients)")
            with st.spinner("Calculating integrated gradients..."):
                attributions = explain_with_captum(clf_model, scaled, target_class=class_idx)
                
                # Sort features by absolute attribution
                feat_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "Attribution": attributions
                })
                # Top 10 important features
                feat_df = feat_df.reindex(feat_df.Attribution.abs().sort_values(ascending=False).index).head(10)
                
                fig_captum = px.bar(
                    feat_df,
                    x="Attribution",
                    y="Feature",
                    orientation="h",
                    title="Top 10 Feature Attributions influencing current classification decision",
                    color="Attribution",
                    color_continuous_scale=px.colors.diverging.RdBu
                )
                fig_captum.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9")
                st.plotly_chart(fig_captum, use_container_width=True)
        else:
            st.info("Captum Integrated Gradients is only compatible with PyTorch deep learning networks. (Current: XGBoost)")
            
        # 2. SHAP Explainer
        st.markdown("#### Local Output Explanation (SHAP Explanation)")
        with st.spinner("Calculating SHAP values..."):
            # Setup a background data for SHAP
            bg_data = np.random.normal(0, 1, size=(25, len(FEATURES)))
            shap_vals = explain_with_shap(clf_model, scaled, bg_data)
            
            # Handle multi-class vs single class output lists
            if isinstance(shap_vals, list):
                # Class specific SHAP
                target_shap = shap_vals[class_idx][0]
            else:
                target_shap = shap_vals[0]
                
            shap_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP Value": target_shap
            })
            shap_df = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index).head(10)
            
            fig_shap = px.bar(
                shap_df,
                x="SHAP Value",
                y="Feature",
                orientation="h",
                title=f"SHAP Values explaining target class: {pred_class}",
                color="SHAP Value",
                color_continuous_scale=px.colors.diverging.Bic
            )
            fig_shap.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9")
            st.plotly_chart(fig_shap, use_container_width=True)

# ==================== TAB 3: MODEL GOVERNANCE ====================
with tab_model:
    st.subheader("🧠 Model Architecture, Performance & Governance Logs")
    st.markdown("Compare model metrics, inspect confusion matrices, and review structural metrics generated during training runs.")
    
    reports_graphs_dir = "D:\\cyber_threat_detection\\reports\\graphs"
    reports_metrics_dir = "D:\\cyber_threat_detection\\reports\\metrics"
    
    col_gov1, col_gov2 = st.columns(2)
    
    with col_gov1:
        st.markdown("#### Anomaly Detection Model Metrics")
        roc_path = os.path.join(reports_graphs_dir, "anomaly_roc.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Autoencoder Anomaly Detection ROC Curve", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate anomaly ROC curves.")
            
        hist_path = os.path.join(reports_graphs_dir, "anomaly_error_hist.png")
        if os.path.exists(hist_path):
            st.image(hist_path, caption="Reconstruction Error Distribution Histogram", use_container_width=True)
            
    with col_gov2:
        st.markdown("#### Attack Classification Performance Comparison")
        comp_path = os.path.join(reports_graphs_dir, "classifier_comparison.png")
        if os.path.exists(comp_path):
            st.image(comp_path, caption="F1 Score vs Inference Latency Comparison of ML Architectures", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate classifier comparison metrics.")
            
        clf_cm_path = os.path.join(reports_graphs_dir, "classifier_confusion.png")
        if os.path.exists(clf_cm_path):
            st.image(clf_cm_path, caption="Confusion Matrix of Best Selected Classifier", use_container_width=True)

# ==================== TAB 4: REPORTS & AUDIT LOGS ====================
with tab_reports:
    st.subheader("📊 Audit Reports & Log Exports")
    st.markdown("Generate downloadable CSV log dumps and download performance summaries for audit tracking.")
    
    # Download current flow logs
    if len(st.session_state.threat_log) > 0:
        log_df = pd.DataFrame(st.session_state.threat_log)
        csv_buffer = io.StringIO()
        log_df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        
        st.markdown("#### Export Current Live Feed Logs")
        st.download_button(
            label="💾 Download CSV Threat Log Dump",
            data=csv_str,
            file_name=f"threat_mitigation_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        st.dataframe(log_df.drop(columns=['raw_features']))
    else:
        st.info("Analyze traffic in Tab 1 to generate downloadable SOC mitigation audit logs.")
        
    st.markdown("---")
    st.markdown("#### Saved Performance Summaries")
    
    summary_path = "D:\\cyber_threat_detection\\reports\\metrics\\summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_data = json.load(f)
        st.json(summary_data)
    else:
        st.info("Summary performance data not found. Complete model training to populate.")
