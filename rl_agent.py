import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from utils import ACTIONS

class CyberSecurityEnv(gym.Env):
    """
    Custom Gymnasium environment for autonomous cybersecurity response.
    
    State Vector (8 dimensions):
    [
        anomaly_score,
        attack_type_encoded,
        confidence_score,
        flow_duration,
        packet_rate,
        threat_severity,
        historical_attack_frequency,
        false_positive_probability
    ]
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, simulation_mode=True, X_eval=None, y_eval=None):
        super(CyberSecurityEnv, self).__init__()
        
        self.simulation_mode = simulation_mode
        self.X_eval = X_eval
        self.y_eval = y_eval
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: 8-dimensional continuous vector
        # Range bounded between 0 and 1 for simplicity of scaled values
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 7, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 100
        self.state = None
        self.current_label = 0
        
    def _generate_state(self):
        """Generates state vectors based on data or simulation."""
        if not self.simulation_mode and self.X_eval is not None:
            # Draw a sample from actual data
            idx = np.random.randint(0, len(self.X_eval))
            sample = self.X_eval[idx]
            label = self.y_eval[idx]
            
            # Formulate state features
            anomaly_score = 0.9 if label != 0 else 0.05
            attack_type_encoded = float(label)
            confidence_score = float(np.random.uniform(0.7, 0.99))
            
            # Normalized values for flow duration and packet rate
            flow_duration = float(np.clip(sample[1] / 10000.0, 0.0, 1.0))
            packet_rate = float(np.clip(sample[15] / 1000.0, 0.0, 1.0))
            
            threat_severity = 0.0 if label == 0 else (0.9 if label in [1, 2, 4, 5] else 0.6)
            historical_attack_frequency = float(np.random.uniform(0.01, 0.2))
            false_positive_probability = float(np.random.uniform(0.01, 0.1))
            
        else:
            # Pure Simulation mode
            label = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], p=[0.7, 0.07, 0.07, 0.04, 0.03, 0.03, 0.03, 0.03])
            anomaly_score = float(np.random.uniform(0.6, 1.0) if label != 0 else np.random.uniform(0.0, 0.2))
            attack_type_encoded = float(label)
            confidence_score = float(np.random.uniform(0.4, 0.99))
            flow_duration = float(np.random.uniform(0.0, 1.0))
            packet_rate = float(np.random.uniform(0.0, 1.0))
            threat_severity = 0.0 if label == 0 else (0.9 if label in [1, 2, 4, 5] else 0.6)
            historical_attack_frequency = float(np.random.uniform(0.0, 0.5))
            false_positive_probability = float(np.random.uniform(0.0, 0.3))
            
        self.current_label = label
        
        state_vec = np.array([
            anomaly_score,
            attack_type_encoded,
            confidence_score,
            flow_duration,
            packet_rate,
            threat_severity,
            historical_attack_frequency,
            false_positive_probability
        ], dtype=np.float32)
        
        return state_vec
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self._generate_state()
        info = {"label_name": self._get_label_name()}
        return self.state, info
        
    def step(self, action):
        self.current_step += 1
        
        # Calculate Reward
        reward = self._calculate_reward(action)
        
        # Next state
        self.state = self._generate_state()
        
        # Terminated / Truncated
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {"label_name": self._get_label_name(), "step": self.current_step}
        
        return self.state, float(reward), terminated, truncated, info
        
    def _get_label_name(self):
        # Maps integer label back to name
        from utils import REV_CLASS_MAP
        return REV_CLASS_MAP.get(int(self.current_label), 'BENIGN')
        
    def _calculate_reward(self, action):
        label = int(self.current_label)
        confidence = self.state[2]
        
        # 1. Benign Traffic Handling
        if label == 0: # BENIGN
            if action == 3: # Ignore (Correct)
                return 10.0
            elif action in [0, 2]: # Block or Quarantine (False Positive Block)
                return -15.0
            elif action in [4]: # Restrict Port (Overreaction)
                return -10.0
            else: # Raise Alert, Monitor Further
                return -5.0 # Mild overreaction
                
        # 2. Critical Attack Handling (DDoS, DoS Hulk, Bot, Infiltration)
        elif label in [1, 2, 4, 5]: 
            if action in [0, 2]: # Block or Quarantine (Correct Mitigation)
                return 10.0
            elif action == 3: # Ignore (Ignore Attack)
                return -20.0
            elif action == 4: # Restrict Port (Partially effective)
                return 5.0
            else: # Raise Alert, Monitor Further
                return -10.0 # Underreaction
                
        # 3. PortScan & Brute Force Handling
        elif label in [3, 7]:
            if action in [4, 0]: # Restrict Port or Block (Correct Mitigation)
                return 10.0
            elif action == 3: # Ignore (Ignore Attack)
                return -20.0
            elif action == 2: # Quarantine (Overreaction)
                return -10.0
            else: # Raise Alert, Monitor Further
                return -10.0 # Underreaction
                
        # 4. Low confidence alert adjustment
        if confidence < 0.6 and action == 1: # Raise Alert on uncertain case
            return 5.0
            
        # Default fallback
        return -5.0

    def render(self):
        print(f"Step {self.current_step}: Class {self._get_label_name()} -> State {self.state}")


def train_rl_agent(base_dir="D:\\cyber_threat_detection", timesteps=10000):
    print("\n--- Training Autonomous Response RL Agent ---")
    
    # Load validation data for environment training if available
    processed_dir = os.path.join(base_dir, "data", "processed")
    X_eval, y_eval = None, None
    if os.path.exists(os.path.join(processed_dir, "X_test.csv")):
        try:
            X_eval = pd.read_csv(os.path.join(processed_dir, "X_test.csv")).values
            y_eval = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.flatten()
        except Exception as e:
            print(f"Could not load evaluation data for RL Env: {e}")
            
    # Instantiate Gymnasium environment
    env = CyberSecurityEnv(simulation_mode=True if X_eval is None else False, X_eval=X_eval, y_eval=y_eval)
    
    # Setup PPO agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log=os.path.join(base_dir, "logs"))
    
    # Checkpoint and save policy directory
    policy_dir = os.path.join(base_dir, "model", "rl_policy")
    os.makedirs(policy_dir, exist_ok=True)
    
    # Train PPO Agent
    model.learn(total_timesteps=timesteps)
    
    # Save the final policy model
    model.save(os.path.join(policy_dir, "ppo_response_agent"))
    print(f"RL policy training completed. Saved policy checkpoint to {policy_dir}")
    
    return model

def load_rl_agent(base_dir="D:\\cyber_threat_detection"):
    policy_path = os.path.join(base_dir, "model", "rl_policy", "ppo_response_agent")
    if os.path.exists(policy_path + ".zip"):
        return PPO.load(policy_path)
    return None

if __name__ == "__main__":
    train_rl_agent(timesteps=15000)
