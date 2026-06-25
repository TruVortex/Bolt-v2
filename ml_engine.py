import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)


class GaitNN(nn.Module):
    """
    Kinematic Neural Network for Running Gait Classification.
    Inputs: [torso_angle, knee_angle, elbow_angle, overstride_ratio]
    Outputs: [
        gait_alignment_score (0.0 to 1.0),
        torso_deviation,      (-1.0 to 1.0; 0.0 is perfect)
        knee_deviation,       (-1.0 to 1.0; 0.0 is perfect)
        elbow_deviation,      (-1.0 to 1.0; 0.0 is perfect)
        overstride_deviation  (0.0 to 1.0; 0.0 is perfect)
    ]
    """

    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.output_layer(x)
        score = self.sigmoid(out[:, 0:1])
        deviations = out[:, 1:]
        return torch.cat((score, deviations), dim=1)


class BoltML:
    def __init__(self, model_path="gait_biomechanics_net.pth"):
        self.model_path = model_path
        self.model = GaitNN()
        self.means = np.array([5.0, 140.0, 95.0, 0.15])
        self.stds = np.array([3.0, 10.0, 15.0, 0.15])
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        else:
            print("No base model found. Compiling baseline.")
            self.train_research_prior_model()

    def normalize_input(self, x):
        """Standardizes input parameters using population statistics."""
        return (x - self.means) / (self.stds + 1e-8)

    def generate_synthetic_research_prior(self, num_samples=1500):
        """
        Generates a synthetic dataset biased heavily toward sports science research limits.
        Identifies normal distributions for elite form and labels outer limits as faulty.
        """
        X = []
        Y = []
        for _ in range(num_samples):
            is_elite = np.random.rand() > 0.6
            if is_elite:
                torso = np.random.normal(5.0, 1.5)
                knee = np.random.normal(140.0, 3.0)
                elbow = np.random.normal(90.0, 5.0)
                overstride = np.random.normal(0.1, 0.05)
                score = np.random.uniform(0.9, 1.0)
                t_dev, k_dev, e_dev, o_dev = 0.0, 0.0, 0.0, 0.0
            else:
                fault_type = np.random.choice(
                    ["upright", "lean", "collapsed_knee", "straight_knee", "stiff_arms", "clamped_arms", "overstride"]
                )
                torso = np.random.normal(5.0, 5.0)
                knee = np.random.normal(140.0, 15.0)
                elbow = np.random.normal(95.0, 25.0)
                overstride = np.random.normal(0.25, 0.2)
                if fault_type == "upright":
                    torso = np.random.normal(-1.0, 1.0)
                elif fault_type == "lean":
                    torso = np.random.normal(13.0, 2.0)
                elif fault_type == "collapsed_knee":
                    knee = np.random.normal(120.0, 3.0)
                elif fault_type == "straight_knee":
                    knee = np.random.normal(160.0, 4.0)
                elif fault_type == "stiff_arms":
                    elbow = np.random.normal(130.0, 10.0)
                elif fault_type == "clamped_arms":
                    elbow = np.random.normal(60.0, 5.0)
                elif fault_type == "overstride":
                    overstride = np.random.normal(0.55, 0.1)
                t_dev = np.clip((torso - 5.0) / 4.0, -1.0, 1.0)
                k_dev = np.clip((knee - 140.0) / 10.0, -1.0, 1.0)
                e_dev = np.clip((elbow - 90.0) / 15.0, -1.0, 1.0)
                o_dev = np.clip(max(0.0, overstride - 0.2) / 0.3, -1.0, 1.0)
                total_error = abs(t_dev) + abs(k_dev) + abs(e_dev) + abs(o_dev)
                score = np.clip(1.0 - (total_error * 0.25), 0.1, 0.85)
            X.append([torso, knee, elbow, overstride])
            Y.append([score, t_dev, k_dev, e_dev, o_dev])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def train_research_prior_model(self, epochs=100):
        """Trains the network on base biomechanical rules."""
        X_raw, Y = self.generate_synthetic_research_prior()
        X = self.normalize_input(X_raw)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X_tensor)
            loss = criterion(predictions, Y_tensor)
            loss.backward()
            optimizer.step()
        print(f"Baseline initialized. MSE Loss: {loss.item():.5f}")
        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()

    def fine_tune_with_athlete_data(self, athlete_features, target_labels, epochs=15, prior_weight=0.5):
        """
        Fine-tunes the network using real professional athlete patterns.
        Utilizes a prior preservation loss (L2 distance from original parameters)
        to prevent catastrophic forgetting of general biomechanical rules.
        """
        base_params = {name: param.clone() for name, param in self.model.named_parameters()}
        X = self.normalize_input(np.array(athlete_features, dtype=np.float32))
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(target_labels, dtype=torch.float32)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.model(X_tensor)
            data_loss = criterion(predictions, Y_tensor)
            prior_loss = 0.0
            for name, param in self.model.named_parameters():
                prior_loss += torch.sum((param - base_params[name]) ** 2)
            total_loss = data_loss + (prior_weight * prior_loss)
            total_loss.backward()
            optimizer.step()
        print(f"Fine-tuning completed. Combined Loss: {total_loss.item():.5f}")
        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()

    def evaluate_gait_features(self, torso, knee, elbow, overstride):
        """Executes inference for a single frame vector."""
        raw_features = np.array([torso, knee, elbow, overstride], dtype=np.float32)
        norm_features = self.normalize_input(raw_features)
        features_tensor = torch.tensor(np.array([norm_features], dtype=np.float32))
        with torch.no_grad():
            outputs = self.model(features_tensor).numpy()[0]
        return {
            "score": round(float(outputs[0]) * 100, 1),
            "torso_dev": float(outputs[1]),
            "knee_dev": float(outputs[2]),
            "elbow_dev": float(outputs[3]),
            "overstride_dev": float(outputs[4])
        }
