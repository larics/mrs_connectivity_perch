import copy
from math import sqrt

import numpy as np
from skimage import measure


class FrontierDetector:
    def __init__(self, occupancy_map, map_resolution, map_origin, robot_size):
        # Initialization remains the same
        self.occupancy_map = copy.deepcopy(occupancy_map)
        self.map_resolution = map_resolution
        self.map_origin = map_origin
        self.r, self.c = occupancy_map.shape
        self.robot_size = robot_size
        self.current_pose = None  # To be set with the robot's current position

    def set_pose(self, pose):
        self.current_pose = pose

    def set_map(self, occupancy_map):
        self.occupancy_map = copy.deepcopy(occupancy_map)

    def detect_frontiers(self):
        """
        Identifies frontier points in the occupancy map (edges between free and unknown space),
        while mitigating noise using morphological operations.

        Returns:
            numpy.ndarray: A binary grid with frontier cells marked.
        """
        # Step 1: Identify frontiers
        frontier_map = np.zeros_like(self.occupancy_map)
        for i in range(1, self.r - 1):
            for j in range(1, self.c - 1):
                if self.occupancy_map[i, j] == 0:  # Free space
                    if -1 in self.occupancy_map[i - 1:i + 2, j - 1:j + 2]:  # Check for unknown space around
                        frontier_map[i, j] = 255  # Mark as a frontier

        # Step 2: Apply morphological operations to remove noise
        # Erode followed by dilation to remove single-pixel noise
        # cleaned_frontier_map = morphology.binary_opening(frontier_map, footprint=morphology.disk(1)).astype(np.uint8) * 255

        # return cleaned_frontier_map
        return frontier_map

    def label_frontiers(self, frontier_map):
        # Label and filter frontiers
        labelled_frontiers = measure.label(frontier_map, background=0)
        regions = measure.regionprops(labelled_frontiers)
        candidate_points = []

        for region in regions:
            if region.area > 9:  # Filter out small noise frontiers
                centroid = region.centroid
                candidate_points.append([int(centroid[0]), int(centroid[1])])

        return candidate_points, labelled_frontiers

    def nearest_frontier(self, candidate_points, state):
        # Implementation remains the same
        distances = []
        for point in candidate_points:
            distance = self.pose_to_grid_distance(point, state)
            distances.append(distance)

        sorted_points = [point for _, point in sorted(zip(distances, candidate_points))]
        sorted_cartesian_points = [self.__map_to_position__(point) for point in sorted_points]

        return np.array(sorted_cartesian_points)

    def pose_to_grid_distance(self, grid_point, pose_point):
        # Implementation remains the same
        world_point = self.__map_to_position__(grid_point)
        return sqrt((pose_point[0] - world_point[0]) ** 2 + (pose_point[1] - world_point[1]) ** 2)

    def __map_to_position__(self, grid_point):
        # Implementation remains the same
        x = grid_point[1] * self.map_resolution + self.map_origin[0]
        y = grid_point[0] * self.map_resolution + self.map_origin[1]
        return [x, y]
