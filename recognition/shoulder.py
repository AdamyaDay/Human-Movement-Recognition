
import numpy as np
from collections import deque
import time


class ShoulderRecognizer:

    def __init__(self):
        self.history = {
            "Left Shoulder": deque(),
            "Right Shoulder": deque()
        }

        self.previous_angles = {
            "Left Shoulder": None,
            "Right Shoulder": None
        }

        self.threshold = 12.0
        self.window = 0.35

    @staticmethod
    def calculate_abduction_angle(landmarks, side):
        if side == "Left":
            shoulder_idx, elbow_idx, hip_idx = 11, 13, 23
            opposite_shoulder_idx = 12
        else:
            shoulder_idx, elbow_idx, hip_idx = 12, 14, 24
            opposite_shoulder_idx = 11

        def point(index):
            lm = landmarks[index]
            return np.array([lm.x, lm.y, lm.z], dtype=float)

        shoulder = point(shoulder_idx)
        elbow = point(elbow_idx)
        hip = point(hip_idx)
        other_shoulder = point(opposite_shoulder_idx)

        # Construct body-relative reference axes.
        lateral = other_shoulder - shoulder
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm < 1e-6:
            return None
        lateral /= lateral_norm

        torso_down = hip - shoulder
        torso_down -= np.dot(torso_down, lateral) * lateral

        torso_norm = np.linalg.norm(torso_down)
        if torso_norm < 1e-6:
            return None
        torso_down /= torso_norm

        upper_arm = elbow - shoulder

        # Project upper arm onto the coronal plane.
        lateral_component = np.dot(upper_arm, lateral)
        down_component = np.dot(upper_arm, torso_down)

        projected_norm = np.hypot(
            lateral_component, down_component
        )
        if projected_norm < 1e-6:
            return None

        angle = np.degrees(
            np.arctan2(
                abs(lateral_component),
                down_component
            )
        )

        return float(angle)

    def update(self, landmarks):
        now = time.monotonic()
        results = {}

        for side in ["Left", "Right"]:
            name = f"{side} Shoulder"
            angle = self.calculate_abduction_angle(
                landmarks, side
            )

            if angle is None:
                self.history[name].clear()
                self.previous_angles[name] = None
                results[name] = {
                    "movement": "Not detected",
                    "angle": None
                }
                continue

            previous = self.previous_angles[name]
            smoothed = (
                angle if previous is None
                else 0.35 * angle + 0.65 * previous
            )
            self.previous_angles[name] = smoothed

            history = self.history[name]
            history.append((now, smoothed))

            while (
                len(history) > 2
                and now - history[0][0] > self.window
            ):
                history.popleft()

            velocity = 0.0
            if len(history) >= 2:
                dt = history[-1][0] - history[0][0]
                if dt > 0:
                    velocity = (
                        history[-1][1] - history[0][1]
                    ) / dt

            if velocity > self.threshold:
                movement = "Abduction"
            elif velocity < -self.threshold:
                movement = "Adduction"
            else:
                movement = "Holding"

            results[name] = {
                "movement": movement,
                "angle": smoothed,
                "velocity": velocity
            }

        return results
