import math

import numpy as np


def get_curve(formation, num_points=5, speed=None, dt=None):
    """
    Initialize agent positions based on the given formation.

    Args:
        formation (list): Formation type and parameters:
                         - ["Circle", [x, y], radius]
                         - ["Elipse", [x, y], major_radius, minor_radius (optional)]
                         - ["Square", [x, y], side_length]
        num_points (int): Number of agents.

    Returns:
        np.ndarray: Array of positions as [[x, y], [x, y], ...].
    """
    formation_type = formation['shape'].lower()
    origin = np.array(formation['origin'])  # Extract the origin

    if speed is not None:
        dist_per_step = speed * dt

    if formation_type == "circle":
        if 'radius' in formation.keys():
            radius = formation['radius']
        if speed is not None:
            num_points = int(2 * math.pi * radius / dist_per_step)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        positions = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])
        return positions + origin  # Offset positions by origin

    elif formation_type == "elipse":
        if 'major_radius' in formation.keys():
            major_radius = formation['major_radius']
        if speed is not None:
            num_points = int(9.68 * major_radius / dist_per_step)
        minor_radius = formation[3] if len(formation) > 3 else major_radius / 2  # Default minor radius
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        positions = np.array([[major_radius * np.cos(angle), minor_radius * np.sin(angle)] for angle in angles])
        return positions + origin  # Offset positions by origin

    elif formation_type == "square":
        if 'side' in formation.keys():
            side_length = formation['side']
            half_side = side_length / 2

        if speed is not None:
            num_points = int(4 * side_length / dist_per_step)

        # Distribute agents along the four sides of the square
        per_side = max(1, num_points // 4)
        remainder = num_points % 4

        # Generate points along each side
        top = np.linspace(-half_side, half_side, per_side + (1 if remainder > 0 else 0), endpoint=False)
        right = np.linspace(half_side, -half_side, per_side + (1 if remainder > 1 else 0), endpoint=False)
        bottom = np.linspace(half_side, -half_side, per_side + (1 if remainder > 2 else 0), endpoint=False)
        left = np.linspace(-half_side, half_side, per_side, endpoint=True)

        # Combine the coordinates for each edge
        positions = np.concatenate([
            np.column_stack((top, np.full_like(top, half_side))),  # Top edge
            np.column_stack((np.full_like(right, half_side), right)),  # Right edge
            np.column_stack((bottom, np.full_like(bottom, -half_side))),  # Bottom edge
            np.column_stack((np.full_like(left, -half_side), left))  # Left edge
        ])

        # Trim to the number of agents
        return positions[:num_points] + origin  # Offset positions by origin

    elif formation_type == 'line':
        # New: Arbitrary line path
        start = np.array(formation.get('origin', [0.0, 0.0]))
        end = np.array(formation.get('end', [1.0, 1.0]))
        num_points = formation.get('num_points', 250)
        path = np.linspace(start, end, num_points)
        return path

    else:
        raise ValueError(f"Unsupported formation type: {formation_type}")


def svstack(arrays):
    """
    Stacks arrays vertically, handling empty arrays by ensuring they have the correct number of columns.

    Parameters:
    - arrays: A list of numpy arrays to stack.

    Returns:
    - A vertically stacked numpy array.
    """
    # Filter out completely empty arrays and get the shape of the first non-empty array
    arrays = [np.array(arr) for arr in arrays]
    non_empty_arrays = [arr for arr in arrays if arr.size > 0]
    if not non_empty_arrays:  # if all arrays are empty, return an empty array
        return np.array([])

    # Assume all non-empty arrays have the same number of columns
    num_columns = non_empty_arrays[0].shape[1] if len(non_empty_arrays[0].shape) > 1 else 0

    # Reshape empty arrays to have the correct number of columns
    arrays = [arr if arr.size > 0 else np.empty((0, num_columns)) for arr in arrays]

    # Stack the arrays vertically
    return np.vstack(arrays)


# Helper function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
