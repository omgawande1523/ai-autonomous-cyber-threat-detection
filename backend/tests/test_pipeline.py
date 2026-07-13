import os
import unittest
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

# Import modules to test
from utils import FEATURES, CLASSES, CLASS_MAP, REV_CLASS_MAP, generate_synthetic_data, download_and_preprocess_dataset
from train import run_pipeline, DenseAutoencoder, MLPClassifier
from rl_agent import CyberSecurityEnv, train_rl_agent
from api import app

class TestCyberThreatSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = "D:\\cyber_threat_detection"
        cls.processed_dir = os.path.join(cls.base_dir, "data", "processed")
        cls.raw_dir = os.path.join(cls.base_dir, "data", "raw")
        
    def test_01_synthetic_data_generation(self):
        """Test if synthetic data generation creates correct features and labels."""
        df = generate_synthetic_data(num_samples=100)
        self.assertEqual(len(df), 100)
        self.assertIn('label', df.columns)
        for feat in FEATURES:
            self.assertIn(feat, df.columns)
            
    def test_02_preprocessing_pipeline(self):
        """Test preprocessing, data scaling, splits, and file saving."""
        total_rows, train_shape, test_shape = download_and_preprocess_dataset(self.base_dir)
        self.assertTrue(total_rows > 0)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(self.processed_dir, "X_train.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.processed_dir, "X_test.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.processed_dir, "y_train.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.processed_dir, "y_test.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.processed_dir, "X_train_benign.csv")))
        
    def test_03_rl_environment_and_agent(self):
        """Test Gym CyberSecurityEnv step mechanics and training loop."""
        env = CyberSecurityEnv(simulation_mode=True)
        state, info = env.reset()
        self.assertEqual(state.shape, (8,))
        self.assertIn("label_name", info)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(0) # Block IP action
        self.assertEqual(next_state.shape, (8,))
        self.assertTrue(isinstance(reward, float))
        
        # Test agent quick training
        model = train_rl_agent(self.base_dir, timesteps=100)
        self.assertTrue(os.path.exists(os.path.join(self.base_dir, "model", "rl_policy", "ppo_response_agent.zip")))

    def test_04_api_prediction(self):
        """Test FastAPI endpoint predictions and model loadings."""
        client = TestClient(app)
        
        # Run startup events manually
        try:
            from api import load_assets
            load_assets()
        except Exception as e:
            print(f"Skipping load assets during test: {e}")
            
        mock_payload = {
            "features": {f: 1.0 for f in FEATURES}
        }
        mock_payload["features"]["flow_duration"] = 120.0
        mock_payload["features"]["flow_packets_s"] = 5.0
        
        # Post predict
        response = client.post("/predict", json=mock_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("anomaly_score", data)
        self.assertIn("is_anomaly", data)
        self.assertIn("attack_type", data)
        self.assertIn("recommended_action", data)

if __name__ == "__main__":
    unittest.main()
