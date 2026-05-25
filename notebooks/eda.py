import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(base_dir="D:\\cyber_threat_detection"):
    print("Starting Exploratory Data Analysis...")
    
    raw_path = os.path.join(base_dir, "data", "raw", "cicids2017_raw.csv")
    reports_graphs_dir = os.path.join(base_dir, "reports", "graphs")
    os.makedirs(reports_graphs_dir, exist_ok=True)
    
    if not os.path.exists(raw_path):
        print("Raw dataset not found. Generating synthetic dataset for analysis...")
        from utils import generate_synthetic_data
        df = generate_synthetic_data()
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)
    else:
        df = pd.read_csv(raw_path)
        
    print(f"Dataset Shape: {df.shape}")
    print("\nClass Distribution:")
    class_counts = df['label'].value_counts()
    print(class_counts)
    
    # Plot 1: Class Distribution Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
    plt.title("CICIDS2017 Dataset Label Distribution")
    plt.xlabel("Traffic Classification")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_graphs_dir, "eda_class_distribution.png"))
    plt.close()
    
    # Plot 2: Correlation Heatmap of top numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Select a subset of features to make heatmap legible
    selected_features = [c for c in ['flow_duration', 'total_fwd_packets', 'total_backward_packets', 
                                    'fwd_packet_length_mean', 'bwd_packet_length_mean', 
                                    'flow_packets_s', 'flow_bytes_s'] if c in df.columns]
    
    if len(selected_features) > 1:
        plt.figure(figsize=(8, 6))
        corr = df[selected_features].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Key Network Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_graphs_dir, "eda_correlation_matrix.png"))
        plt.close()
        
    print("EDA completed! Distribution and correlation plots saved in reports/graphs/.")

if __name__ == "__main__":
    run_eda()
