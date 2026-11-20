
from collections import deque
import time


class MovementRecognizer:
    """
    Rule-based anatomical movement recognition.

    Uses changes in joint angles over time to identify
    flexion and extension.
    """

    def __init__(
        self,
        velocity_threshold=12.0,
        history_duration=0.35,
        smoothing_factor=0.35
    ):
        self.velocity_threshold = velocity_threshold
        self.history_duration = history_duration
        self.smoothing_factor = smoothing_factor

        self.joints = [
            "Left Elbow",
            "Right Elbow",
            "Left Knee",
            "Right Knee"
        ]

        self.history = {
            joint: deque()
            for joint in self.joints
        }

        self.smoothed_angles = {
            joint: None
            for joint in self.joints
        }

        self.states = {
            joint: "Initializing"
            for joint in self.joints
        }

    def update(self, angles):
        """
        Update movement classifications using
        the latest joint angles.

        Returns:
            Dictionary containing movement labels
            and angular velocities.
        """

        current_time = time.monotonic()

        results = {}

        for joint in self.joints:

            angle = angles.get(joint)

            if angle is None:
                results[joint] = {
                    "movement": "Not detected",
                    "velocity": 0.0
                }
                continue

            # --------------------------------
            # 1. Smooth angle measurements
            # --------------------------------

            previous = self.smoothed_angles[joint]

            if previous is None:
                smoothed = angle
            else:
                smoothed = (
                    self.smoothing_factor * angle
                    + (1 - self.smoothing_factor) * previous
                )

            self.smoothed_angles[joint] = smoothed

            # --------------------------------
            # 2. Store angle history
            # --------------------------------

            history = self.history[joint]

            history.append((current_time, smoothed))

            # Remove old measurements
            while (
                history
                and current_time - history[0][0]
                > self.history_duration
            ):
                history.popleft()

            # --------------------------------
            # 3. Calculate angular velocity
            # --------------------------------

            if len(history) < 2:
                movement = "Initializing"
                velocity = 0.0

            else:
                first_time, first_angle = history[0]
                last_time, last_angle = history[-1]

                elapsed = last_time - first_time

                if elapsed <= 0:
                    velocity = 0.0
                    movement = "Holding"

                else:
                    velocity = (
                        (last_angle - first_angle)
                        / elapsed
                    )

                    # --------------------------------
                    # 4. Classify movement
                    # --------------------------------

                    if velocity < -self.velocity_threshold:
                        movement = "Flexion"

                    elif velocity > self.velocity_threshold:
                        movement = "Extension"

                    else:
                        movement = "Holding"

            self.states[joint] = movement

            results[joint] = {
                "movement": movement,
                "velocity": velocity,
                "angle": smoothed
            }

        return results
