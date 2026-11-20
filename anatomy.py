
import numpy as np


def calculate_3d_angle(point_a, point_b, point_c):
    """
    Calculate the 3D angle at point B.

    A, B, C are 3D coordinates.
    Returns an angle in degrees.
    """

    a = np.array(point_a, dtype=float)
    b = np.array(point_b, dtype=float)
    c = np.array(point_c, dtype=float)

    vector_ba = a - b
    vector_bc = c - b

    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine_angle = np.dot(vector_ba, vector_bc) / (
        norm_ba * norm_bc
    )

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))

    return float(angle)


def get_landmark_coordinates(landmarks, index):
    """
    Extract estimated 3D world coordinates
    from a MediaPipe landmark.
    """

    landmark = landmarks[index]

    return (
        landmark.x,
        landmark.y,
        landmark.z
    )


def calculate_joint_angles(landmarks):
    """
    Calculate selected joint angles from
    MediaPipe world landmarks.
    """

    # MediaPipe landmark indices
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15

    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16

    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27

    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28

    angles = {}

    # Left elbow
    angles["Left Elbow"] = calculate_3d_angle(
        get_landmark_coordinates(landmarks, LEFT_SHOULDER),
        get_landmark_coordinates(landmarks, LEFT_ELBOW),
        get_landmark_coordinates(landmarks, LEFT_WRIST)
    )

    # Right elbow
    angles["Right Elbow"] = calculate_3d_angle(
        get_landmark_coordinates(landmarks, RIGHT_SHOULDER),
        get_landmark_coordinates(landmarks, RIGHT_ELBOW),
        get_landmark_coordinates(landmarks, RIGHT_WRIST)
    )

    # Left knee
    angles["Left Knee"] = calculate_3d_angle(
        get_landmark_coordinates(landmarks, LEFT_HIP),
        get_landmark_coordinates(landmarks, LEFT_KNEE),
        get_landmark_coordinates(landmarks, LEFT_ANKLE)
    )

    # Right knee
    angles["Right Knee"] = calculate_3d_angle(
        get_landmark_coordinates(landmarks, RIGHT_HIP),
        get_landmark_coordinates(landmarks, RIGHT_KNEE),
        get_landmark_coordinates(landmarks, RIGHT_ANKLE)
    )

    return angles
