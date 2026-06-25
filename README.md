---
title: Bolt
emoji: 🏃
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# [Bolt-v2](https://huggingface.co/spaces/TruVortex/Bolt)

A reimagination of [a previous project](https://github.com/MSS-Goats/Bolt), Bolt is a biomechanics-informed gait analysis project that uses computer vision, feature engineering, and machine learning to evaluate running form from video. The app processes uploaded side-profile video, isolates a complete running stride of the prominent leg, and provides relavent feedback.

---

## System Architecture
```
Video Input
      ↓
MediaPipe Pose Estimation
      ↓
Landmark Tracking
      ↓
Stride Segmentation
      ↓
Biomechanical Feature Extraction
      ↓
Multi-Task Neural Network
      ↓
Running Form Assessment
```

### Multi-Task Learning
Rather than executing multiple disconnected models or rule-based heuristics, Bolt uses a single multi-output neural network:
* **Multi-Task Learning (MTL):** The model simultaneously predicts an overall `gait_score` (continuous 0.0 to 1.0 regression) and four localized joint deviations (`torso_dev`, `knee_dev`, `elbow_dev`, and `overstride_dev`). 
* **Gradient Regularization:** Because the overall score represents the global integration of local joint states, training them jointly forces the shared hidden layers (`fc1` and `fc2`) to learn a cohesive representation of biomechanics. This prevents the network from learning contradictory configurations (e.g., predicting zero joint deviations but an extremely low score).
* **Prior-Regularized Training:** To overcome data scarcity and prevent overfitting to specific athlete proportions, the model is initialized on a synthetic distribution of 1,500 samples based on sports-science research, anchoring it to established biomechanical principles. The pretrained model is subsequently adapted using real athlete data (Van Hooren et al. OSF dataset) while using L2 weight regularization to prevent forgetting.

### Stride Segmentation
In cases where a video has slow-motion segments, variable frame rates, or frame drops, standard frame-slicing algorithms can fail. Due to the time-importance of each frame-slice, Bolt uses a purely geometric, state-space approach:
* **Temporal Pelvic Continuity:** MediaPipe’s left/right joint labels frequently swap at gait boundary crossovers due to tracking occlusion. Bolt tracks the 2D frame-to-frame coordinate distance of the hips. Since the hips are physically separated on the pelvis and never cross over, their relative distance acts as an anchor. If a sudden spatial jump is detected, the engine instantly swaps the limb labels back to maintain temporal tracking consistency.
* **The Nose-Orientation Rule:** To determine if a peak in ankle-to-hip displacement represents a forward extension (Initial Contact) or backward extension (Toe-Off), Bolt enforces the anatomical constraint:
  $$\text{sign}(x_{\text{ankle}} - x_{\text{hip}}) == \text{sign}(x_{\text{nose}} - x_{\text{hip}})$$
  Since the nose always points in the direction of forward motion, this check is immune to camera mirroring or running directions.
* **Geometric Bisection:** Once the start and end foot strikes of the prominent leg are located, the intermediate frames are found by searching for specific geometric states, making the sampler immune to selective slow-motion stretching.

---

## Gait Phases

The engine segments and evaluates the prominent leg through 8 distinct, phases:

| Phase        | Phase Label              | Biomechanical Target Evaluated |
|:-------------|:-------------------------| :--- |
| **1 (0%)**   | **Initial Contact**      | Heel Strike overstride ratio relative to pelvic center |
| **2 (14%)**  | **Loading Response**     | Control and soft cushion of landing knee flexion |
| **3 (28%)**  | **Mid-Stance**           | Weight-bearing support-knee flexion (shock absorption) |
| **4 (43%)**  | **Terminal Stance**      | Pelvic and torso posture alignment under peak load |
| **5 (57%)**  | **Toe-Off**              | Kinetic leg extension and ankle push-off drive |
| **6 (71%)**  | **Initial Swing**        | Swing-phase heel recovery and knee drive height |
| **7 (85%)**  | **Terminal Swing**       | Active hamstring deceleration preparing for contact |
| **8 (100%)** | **Next Initial Contact** | Stride completion boundary verification |

---

## Setup & Installation

Bolt was built in Python 3.10 and uses `uv` for dependency management. 

### 1. Synchronize the Environment
Ensure you have `uv` installed. Run the following command to sync your environment with the project's lockfile:

```bash
# Syncs the virtual environment to match uv.lock
uv sync
```

### 2. Compile Biomechanical Data & Train Model
Before launching, you must transform the raw OpenSim Inverse Kinematics (`.mot`) files into an aligned training format. Use the included ETL pipeline to pre-train the model on sports-science priors, followed by fine-tuning on elite athlete data:

```bash
# 1. Parse and align subject files from /osf_data
uv run curate_osf.py

# 2. Pre-train on research-based priors & fine-tune on elite kinematic data
uv run train_pipeline.py
```

### 3. Launch the Application
Start the Flask development server using the managed environment:

```bash
uv run app.py
```

Access the interface at `http://127.0.0.1:5000`.
