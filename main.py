
import cv2
import mediapipe as mp
import time

from anatomy import (
    calculate_joint_angles,
    get_landmark_coordinates
)

from recognition.rule_based import MovementRecognizer
from recognition.shoulder import ShoulderRecognizer


# ==========================================
# 1. MEDIAPIPE INITIALIZATION
# ==========================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# 2. RECOGNITION ENGINE INITIALIZATION
# ==========================================

movement_recognizer = MovementRecognizer(
    velocity_threshold=12.0,
    history_duration=0.35,
    smoothing_factor=0.35
)

shoulder_recognizer = ShoulderRecognizer()


# ==========================================
# 3. WEBCAM INITIALIZATION
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    pose.close()
    raise SystemExit("ERROR: Webcam unavailable.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

previous_time = time.time()


print("==========================================")
print("   3D ANATOMICAL MOVEMENT ANALYSIS SYSTEM")
print("==========================================")
print("MediaPipe Pose Estimation: ACTIVE")
print("3D Landmark Extraction: ACTIVE")
print("Flexion/Extension Recognition: ACTIVE")
print("Abduction/Adduction Recognition: ACTIVE")
print("Press Q to exit.")
print("==========================================")


# ==========================================
# 4. MAIN PROCESSING LOOP
# ==========================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("ERROR: Unable to read camera frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Process pose
        results = pose.process(rgb_frame)

        # ==================================
        # 5. BODY DETECTION
        # ==================================

        if (
            results.pose_landmarks
            and results.pose_world_landmarks
        ):

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            world_landmarks = (
                results.pose_world_landmarks.landmark
            )

            # ==================================
            # 6. JOINT ANGLE CALCULATION
            # ==================================

            angles = calculate_joint_angles(
                world_landmarks
            )

            # ==================================
            # 7. FLEXION / EXTENSION RECOGNITION
            # ==================================

            movement_results = movement_recognizer.update(
                angles
            )

            # ==================================
            # 8. ABDUCTION / ADDUCTION RECOGNITION
            # ==================================

            shoulder_results = shoulder_recognizer.update(
                world_landmarks
            )

            # Combine both recognition results
            movement_results.update(shoulder_results)

            # ==================================
            # 9. DISPLAY SYSTEM STATUS
            # ==================================

            cv2.putText(
                frame,
                "3D ANATOMICAL MOVEMENT ANALYSIS",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "HUMAN DETECTED",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # ==================================
            # 10. DISPLAY MOVEMENT RECOGNITION
            # ==================================

            y_position = 105

            cv2.putText(
                frame,
                "MOVEMENT RECOGNITION",
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            y_position += 35

            for joint, data in movement_results.items():

                angle = data.get("angle")
                movement = data["movement"]
                velocity = data.get("velocity", 0.0)

                # Angle display
                if angle is None:
                    angle_text = "N/A"
                else:
                    angle_text = f"{angle:.1f} deg"

                # Movement colour
                if movement in ["Flexion", "Abduction"]:
                    color = (0, 255, 0)

                elif movement in ["Extension", "Adduction"]:
                    color = (255, 180, 0)

                elif movement == "Holding":
                    color = (200, 200, 200)

                else:
                    color = (0, 165, 255)

                # Joint name and angle
                cv2.putText(
                    frame,
                    f"{joint}: {angle_text}",
                    (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1
                )

                y_position += 23

                # Movement classification
                cv2.putText(
                    frame,
                    f"  Movement: {movement}",
                    (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2
                )

                y_position += 28

            # ==================================
            # 11. DISPLAY 3D COORDINATES
            # ==================================

            coordinate_y = max(y_position + 10, 430)

            cv2.putText(
                frame,
                "LEFT ARM 3D COORDINATES",
                (20, coordinate_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                2
            )

            coordinate_y += 27

            coordinate_points = [
                ("Shoulder", 11),
                ("Elbow", 13),
                ("Wrist", 15)
            ]

            for name, landmark_id in coordinate_points:

                point = get_landmark_coordinates(
                    world_landmarks,
                    landmark_id
                )

                text = (
                    f"{name}: "
                    f"X:{point[0]:.2f} "
                    f"Y:{point[1]:.2f} "
                    f"Z:{point[2]:.2f}"
                )

                cv2.putText(
                    frame,
                    text,
                    (20, coordinate_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

                coordinate_y += 23

        else:

            # ==================================
            # 12. NO BODY DETECTED
            # ==================================

            cv2.putText(
                frame,
                "NO HUMAN DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ==================================
        # 13. FPS CALCULATION
        # ==================================

        current_time = time.time()

        fps = 1.0 / max(
            current_time - previous_time,
            0.0001
        )

        previous_time = current_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (frame.shape[1] - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ==================================
        # 14. DISPLAY WINDOW
        # ==================================

        cv2.imshow(
            "3D Anatomical Movement Analysis",
            frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# ==========================================
# 15. RESOURCE CLEANUP
# ==========================================

finally:

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    print("==========================================")
    print("Camera released.")
    print("Application closed.")
    print("==========================================")
