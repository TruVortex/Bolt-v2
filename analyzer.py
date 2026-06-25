import cv2
import mediapipe as mp
import numpy as np

from ml_engine import BoltML


class RunningAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.ml_pipeline = BoltML()

    @staticmethod
    def calculate_angle_2d(a, b, c):
        """Calculates the inner 2D angle between three points (b is vertex) in isotropic pixel space."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return round(float(np.degrees(angle)), 1)

    def analyze_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_idx = 0
        landmarks_history = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks and results.pose_world_landmarks:
                lms_2d = {i: [lm.x, lm.y] for i, lm in enumerate(results.pose_landmarks.landmark)}
                lms_3d = {i: [lm.x, lm.y, lm.z] for i, lm in enumerate(results.pose_world_landmarks.landmark)}
                visibilities = {i: lm.visibility for i, lm in enumerate(results.pose_landmarks.landmark)}
                if len(landmarks_history) > 0:
                    prev_lms = landmarks_history[-1]['landmarks']
                    if lms_2d[0] == prev_lms[0] and lms_2d[11] == prev_lms[11]:
                        break
                landmarks_history.append(
                    {
                        'frame_idx': frame_idx,
                        'landmarks': lms_2d,
                        'world_landmarks': lms_3d,
                        'visibilities': visibilities
                    }
                )
            frame_idx += 1
        cap.release()
        total_frames = len(landmarks_history)
        if total_frames < 10:
            raise ValueError("Insufficient tracking frame length or no pose detected.")

        left_vis_avg = np.mean(
            [f['visibilities'][23] + f['visibilities'][25] + f['visibilities'][27] for f in landmarks_history]
        )
        right_vis_avg = np.mean(
            [f['visibilities'][24] + f['visibilities'][26] + f['visibilities'][28] for f in landmarks_history]
        )
        closer_side = 'left' if left_vis_avg > right_vis_avg else 'right'

        is_left = closer_side == 'left'
        shoulder_idx = 11 if is_left else 12
        elbow_idx = 13 if is_left else 14
        wrist_idx = 15 if is_left else 16
        hip_idx = 23 if is_left else 24
        knee_idx = 25 if is_left else 26
        ankle_idx = 27 if is_left else 28
        heel_idx = 29 if is_left else 30

        rel_x_history = []
        for f in landmarks_history:
            prominent_ankle = f['landmarks'][ankle_idx]
            prominent_hip = f['landmarks'][hip_idx]
            rel_x_history.append(prominent_ankle[0] - prominent_hip[0])
        smoothed_rel_x = []
        k_size = 7
        for i in range(len(rel_x_history)):
            start = max(0, i - k_size // 2)
            end = min(len(rel_x_history), i + k_size // 2 + 1)
            smoothed_rel_x.append(np.mean(rel_x_history[start:end]))

        nose_to_hip_x = np.mean([f['landmarks'][0][0] - f['landmarks'][hip_idx][0] for f in landmarks_history])
        facing_direction_sign = 1.0 if nose_to_hip_x > 0 else -1.0
        threshold = 0.015
        half_waves = []
        current_type = None
        start_idx = 0
        for i in range(len(smoothed_rel_x)):
            val = smoothed_rel_x[i]
            t = "forward" if (val * facing_direction_sign) > threshold else "backward"
            if t != current_type:
                if current_type is not None:
                    half_waves.append({"type": current_type, "start": start_idx, "end": i - 1})
                current_type = t
                start_idx = i
        half_waves.append({"type": current_type, "start": start_idx, "end": len(smoothed_rel_x) - 1})
        forward_peaks = []
        for hw in half_waves:
            if hw["type"] == "forward":
                slice_data = smoothed_rel_x[hw["start"]: hw["end"] + 1]
                if len(slice_data) > 0:
                    local_idx = hw["start"] + (
                        np.argmax(slice_data) if facing_direction_sign > 0 else np.argmin(slice_data))
                    forward_peaks.append(local_idx)

        is_stride_valid = False
        if len(forward_peaks) >= 2:
            p0, p2 = forward_peaks[0], forward_peaks[1]
            if (p2 - p0) >= int(fps * 0.45):
                is_stride_valid = True
        if is_stride_valid:
            start_idx = p0
            end_idx = p2
        else:
            start_idx = int(total_frames * 0.08)
            end_idx = int(total_frames * 0.92)

        f1_idx = start_idx
        f5_idx = start_idx + (np.argmin(smoothed_rel_x[start_idx:end_idx]) if facing_direction_sign > 0 else np.argmax(
            smoothed_rel_x[start_idx:end_idx]
        ))
        stance_slice = smoothed_rel_x[start_idx:f5_idx]
        f3_idx = start_idx + np.argmin(np.abs(stance_slice))
        f2_idx = (f1_idx + f3_idx) // 2
        f4_idx = (f3_idx + f5_idx) // 2
        f6_idx = f5_idx
        min_knee_val = 180.0
        for i in range(f5_idx, end_idx):
            f_data = landmarks_history[i]['landmarks']
            pixel_h = [f_data[hip_idx][0] * width, f_data[hip_idx][1] * height]
            pixel_k = [f_data[knee_idx][0] * width, f_data[knee_idx][1] * height]
            pixel_a = [f_data[ankle_idx][0] * width, f_data[ankle_idx][1] * height]
            k_angle = self.calculate_angle_2d(pixel_h, pixel_k, pixel_a)
            if k_angle < min_knee_val:
                min_knee_val = k_angle
                f6_idx = i
        f8_idx = end_idx
        f7_idx = (f6_idx + f8_idx) // 2
        sampled_indices = [f1_idx, f2_idx, f3_idx, f4_idx, f5_idx, f6_idx, f7_idx, f8_idx]
        for idx in range(1, len(sampled_indices)):
            if sampled_indices[idx] <= sampled_indices[idx - 1]:
                sampled_indices[idx] = min(total_frames - 1, sampled_indices[idx - 1] + 1)

        sampled_frames = [landmarks_history[idx] for idx in sampled_indices]
        pairs = [
            (11, 12), (13, 14), (15, 16), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
        ]
        for f in sampled_frames:
            lms_2d = f['landmarks']
            lms_3d = f['world_landmarks']
            left_hip_z = lms_3d[23][2]
            right_hip_z = lms_3d[24][2]
            should_swap = False
            if closer_side == 'left' and (left_hip_z > right_hip_z + 0.05):
                should_swap = True
            elif closer_side == 'right' and (right_hip_z > left_hip_z + 0.05):
                should_swap = True
            if should_swap:
                for p1, p2 in pairs:
                    lms_2d[p1], lms_2d[p2] = lms_2d[p2], lms_2d[p1]
                    lms_3d[p1], lms_3d[p2] = lms_3d[p2], lms_3d[p1]

        analysis_payload = []
        phases = [
            "Initial Contact (Heel Strike)",
            "Loading Response (Weight Shift)",
            "Mid-Stance (Max Support)",
            "Terminal Stance (Push-Off)",
            "Toe-Off (Propulsion)",
            "Initial Swing (Leg Drive)",
            "Terminal Swing (Preparation)",
            "Next Initial Contact"
        ]

        for idx, f in enumerate(sampled_frames):
            lms_2d = f['landmarks']
            pixel_coords = {
                i: [coords[0] * width, coords[1] * height]
                for i, coords in lms_2d.items()
            }
            hip_px = pixel_coords[hip_idx]
            shoulder_px = pixel_coords[shoulder_idx]
            vertical_px = [hip_px[0], hip_px[1] - 100]

            torso_angle = self.calculate_angle_2d(vertical_px, hip_px, shoulder_px)
            knee_angle = self.calculate_angle_2d(pixel_coords[hip_idx], pixel_coords[knee_idx], pixel_coords[ankle_idx])
            elbow_angle = self.calculate_angle_2d(
                pixel_coords[shoulder_idx], pixel_coords[elbow_idx], pixel_coords[wrist_idx]
            )

            heel_2d = lms_2d[heel_idx]
            hip_2d = lms_2d[hip_idx]
            sh_2d = lms_2d[shoulder_idx]
            torso_height_norm = np.linalg.norm(np.array(sh_2d) - np.array(hip_2d))
            overstride_ratio = (heel_2d[0] - hip_2d[0]) / torso_height_norm if torso_height_norm > 0 else 0
            if closer_side == 'left':
                overstride_ratio = -overstride_ratio

            eval_metrics = self.ml_pipeline.evaluate_gait_features(
                torso_angle, knee_angle, elbow_angle, overstride_ratio
            )
            advice = self.generate_advice(phases[idx], eval_metrics)
            landmarks_payload = {
                "nose": lms_2d[0],
                "l_shoulder": lms_2d[11],
                "r_shoulder": lms_2d[12],
                "l_elbow": lms_2d[13],
                "r_elbow": lms_2d[14],
                "l_wrist": lms_2d[15],
                "r_wrist": lms_2d[16],
                "l_hip": lms_2d[23],
                "r_hip": lms_2d[24],
                "l_knee": lms_2d[25],
                "r_knee": lms_2d[26],
                "l_ankle": lms_2d[27],
                "r_ankle": lms_2d[28],
                "l_heel": lms_2d[29],
                "r_heel": lms_2d[30],
                "l_toe": lms_2d[31],
                "r_toe": lms_2d[32],
            }
            analysis_payload.append(
                {
                    "phase_name": phases[idx],
                    "timestamp": f['frame_idx'] / fps,
                    "torso_angle": torso_angle,
                    "knee_angle": knee_angle,
                    "elbow_angle": elbow_angle,
                    "overstride_ratio": round(overstride_ratio, 2),
                    "advice": advice,
                    "landmarks": landmarks_payload
                }
            )
        return {
            "side": closer_side,
            "video_metadata": {
                "width": width,
                "height": height
            },
            "frames": analysis_payload
        }

    def generate_advice(self, phase, eval_metrics):
        bullets = []
        t_dev = eval_metrics["torso_dev"]
        k_dev = eval_metrics["knee_dev"]
        e_dev = eval_metrics["elbow_dev"]
        o_dev = eval_metrics["overstride_dev"]
        if "Initial Contact" in phase:
            if o_dev > 0.4:
                bullets.append(
                    {
                        "level": "CRITICAL",
                        "priority": 1,
                        "text": "Overstriding vector detected. Your landing heel is striking too far in front of your vertical hip axis. Focus on increasing stride cadence to land closer to your center of mass."
                    }
                )
            else:
                bullets.append(
                    {
                        "level": "OPTIMAL",
                        "priority": 3,
                        "text": "Excellent landing alignment. Your foot strikes close beneath your center of mass."
                    }
                )
        if "Loading Response" in phase:
            if k_dev < -0.3:
                bullets.append(
                    {
                        "level": "WARNING",
                        "priority": 2,
                        "text": "Your landing leg is too stiff. Ensure a soft, controlled bend in your landing knee during loading to disperse impact forces."
                    }
                )
            else:
                bullets.append(
                    {
                        "level": "OPTIMAL",
                        "priority": 3,
                        "text": "Good landing shock absorption. Joint flex is well-tuned."
                    }
                )
        if "Mid-Stance" in phase:
            if k_dev < -0.4:
                bullets.append(
                    {
                        "level": "CRITICAL",
                        "priority": 1,
                        "text": "Severe loading knee collapse. The neural network detects high support-leg knee compression. Work on single-leg stability to secure force distribution."
                    }
                )
            elif k_dev > 0.4:
                bullets.append(
                    {
                        "level": "WARNING",
                        "priority": 2,
                        "text": "Your support knee is too straight. Ensure a soft bend in the joint to actively absorb impact forces."
                    }
                )
            else:
                bullets.append(
                    {
                        "level": "OPTIMAL",
                        "priority": 3,
                        "text": "Optimal support knee flexion. Solid shock absorption."
                    }
                )
        if "Terminal Stance" in phase or "Toe-Off" in phase:
            if k_dev < -0.3:
                bullets.append(
                    {
                        "level": "WARNING",
                        "priority": 2,
                        "text": "Incomplete leg extension. Ensure full push-off drive through your hip and knee joints to generate maximum forward propulsion."
                    }
                )
            else:
                bullets.append(
                    {
                        "level": "OPTIMAL",
                        "priority": 3,
                        "text": "Excellent extension. Generating optimal power out of your stride push-off."
                    }
                )
        if "Initial Swing" in phase:
            if k_dev > 0.4:
                bullets.append(
                    {
                        "level": "WARNING",
                        "priority": 2,
                        "text": "Low knee drive during swing. Your recovery leg is too straight, dragging behind. Tuck your heel closer to your glutes for a more compact forward drive."
                    }
                )
            else:
                bullets.append(
                    {
                        "level": "OPTIMAL",
                        "priority": 3,
                        "text": "Strong leg recovery. Excellent knee drive and compact heel tuck."
                    }
                )
        if t_dev < -0.7:
            bullets.append(
                {
                    "level": "WARNING",
                    "priority": 2,
                    "text": "Your torso posture is leaning backward or completely upright. Introduce a subtle ankle-initiated forward lean."
                }
            )
        elif t_dev > 0.3:
            bullets.append(
                {
                    "level": "WARNING",
                    "priority": 2,
                    "text": "Excessive upper body forward lean. This restricts hip flexor extension and shortens back-kick clearance."
                }
            )
        else:
            bullets.append(
                {
                    "level": "OPTIMAL",
                    "priority": 3,
                    "text": "Slight forward torso lean is well-aligned, permitting optimized ground force production."
                }
            )
        if e_dev > 0.4:
            bullets.append(
                {
                    "level": "CRITICAL",
                    "priority": 1,
                    "text": "Excessive arm drive extension. Keeping your elbow angle open creates a wider pendulum. Shorten your arm lever to keep transitions clean."
                }
            )
        elif e_dev < -0.3:
            bullets.append(
                {
                    "level": "WARNING",
                    "priority": 2,
                    "text": "Your elbow joint is closed too tightly. Relax your shoulders and open your arms slightly to maintain comfortable forward-back swing planes."
                }
            )
        else:
            bullets.append(
                {
                    "level": "OPTIMAL",
                    "priority": 3,
                    "text": "Excellent arm carriage. Your elbow flexion sits near the optimal 90-degree threshold."
                }
            )
        bullets.sort(key=lambda x: x['priority'])
        return bullets
