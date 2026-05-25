# GitHub Demo Video Showcase Script
**Project**: AI-Powered Autonomous Cyber Threat Detection & Response System

**Target Video Duration**: 3 minutes  
**Format**: Screen capture of Streamlit UI and terminal with voiceover.

---

## Scene 1: Introduction (0:00 - 0:45)
* **Visual**: Show the dark-themed Streamlit SOC Command Center dashboard header with zero metrics.
* **Audio**:
  > "Welcome to the showcase of the AI-Powered Autonomous Cyber Threat Detection and Response System. This is a research-grade security operations dashboard designed to monitor network flows, classify active exploits, and autonomously mitigate threats using Reinforcement Learning."
  > "On the sidebar, we can select our threat monitoring mode—either Simulated flow streams or Live Scapy network sniffing—and view model specifications. Let's start the Threat Monitor."
* **Action**: Click "Start Threat Monitoring" on the sidebar. Show the dashboard metrics (Analyzed Packets, Intercepted Threats) beginning to increment in real-time.

---

## Scene 2: Live Ops Feed (0:45 - 1:30)
* **Visual**: Focus on the real-time Plotly charts updating. Point at the Autoencoder Reconstruction Error timeline and the threat categories distribution.
* **Audio**:
  > "As packets are captured, they are cleaned and scaled. The system first forwards them to our unsupervised LSTM Autoencoder. If reconstruction error exceeds our computed threshold (indicated by the dashed red line), it flags an anomaly."
  > "Simultaneously, our XGBoost classifier predicts the threat type. On the right, our incident mitigation log feed displays active alerts. You can see our trained PPO reinforcement learning agent autonomously choosing response mitigations—like blocking IPs or quarantining devices—minimizing human analyst overhead."

---

## Scene 3: Threat Simulator & XAI (1:30 - 2:30)
* **Visual**: Switch to **Tab 2: Threat Simulator**. Input custom packet values and click "Simulate & Analyze Flow". Scroll down to the Captum and SHAP charts.
* **Audio**:
  > "To validate the system, we can switch to the Threat Simulator tab. Here, security analysts can manually inject custom flow statistics—such as long durations, high packet rates, or SYN flags."
  > "Once analyzed, the system returns the anomaly score, classifier prediction, confidence, and RL agent mitigation recommendation."
  > "To ensure trustworthiness, the Explainable AI panel uses SHAP and Captum Integrated Gradients. This displays the precise feature attributions driving the model's classification decision, turning a black-box model into a fully transparent security tool."

---

## Scene 4: Code, API & Reports (2:30 - 3:00)
* **Visual**: Switch to **Tab 3** and **Tab 4** (Governance and Reports) to show saved graphs and CSV logs.
* **Audio**:
  > "Under Model Governance, we can track cross-validation ROC curves and confusion matrices. Under Reports and Audits, analysts can download complete threat log history as CSV files for compliance tracking."
  > "The system is fully containerized with Docker and exposes a FastAPI endpoint `/predict` for external SIEM integration."
  > "Thanks for watching! Check out the repository README for installation guidelines and the complete research paper."
