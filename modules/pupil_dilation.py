def calculate_pupil_diameter(landmarks, iris_indices, euclidean):
    """
    Estimate pupil (or iris) diameter from the refined iris landmarks.
    For the left iris, use indices 468 and 472 (these are typically the leftmost and rightmost points).
    """
    left_point = landmarks[iris_indices[0]]  # e.g., index 468
    right_point = landmarks[iris_indices[2]]  # e.g., index 472
    return euclidean(left_point, right_point)
