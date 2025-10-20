import copy
from math import sqrt

import numpy as np
from skimage import measure

"""
FrontierDetector identifies frontier points in an occupancy map. 
This class is used in robotic exploration tasks to identify frontiers, which are 
regions separating explored (free) space from unexplored (unknown) space. It includes 
methods for detecting frontiers, labeling them, and selecting the nearest frontier 
relative to the robot's position.
"""


class FrontierDetector:
    def __init__(self, occupancy_map, map_resolution, map_origin, robot_size):
        self.occupancy_map = copy.deepcopy(occupancy_map)
        self.map_resolution = map_resolution
        self.map_origin = map_origin
        self.r, self.c = occupancy_map.shape
        self.robot_size = robot_size
        self.current_pose = None  # To be set with the robot's current position

    def detect_frontiers(self):
        """
        Identifies frontier points in the occupancy map.

        A frontier point is a cell that is free (0) and adjacent to an unknown (-1) cell.
        This method uses a sliding window to find such points and returns a binary grid
        with frontiers marked.

        Returns:
            numpy.ndarray: A binary grid (same shape as the occupancy map) where
                           frontier cells are marked with 255.
        """

        # Step 1: Identify frontiers
        frontier_map = np.zeros_like(self.occupancy_map)
        for i in range(1, self.r - 1):
            for j in range(1, self.c - 1):
                if self.occupancy_map[i, j] == 0:  # Free space
                    # Check for unknown space in the 3x3 neighborhood
                    if -1 in self.occupancy_map[i - 1:i + 2, j - 1:j + 2]:  # Check for unknown space around
                        frontier_map[i, j] = 255  # Mark as a frontier

        # Step 2: Apply morphological operations to remove noise
        # Erode followed by dilation to remove single-pixel noise
        # cleaned_frontier_map = morphology.binary_opening(frontier_map, footprint=morphology.disk(1)).astype(np.uint8) * 255

        return frontier_map

    def label_frontiers(self, frontier_map):
        """
        Labels connected frontier regions and filters out small noise.
        Uses connected component labeling to group frontier cells into regions and
        filters out regions smaller than a specified area threshold.

        Args:
            frontier_map (numpy.ndarray): Binary grid with frontiers marked.

        Returns:
            tuple:
                - candidate_points (list): List of centroid positions of labeled frontiers.
                - labelled_frontiers (numpy.ndarray): Grid with labeled frontier regions.
        """
        # Label connected frontier regions
        labelled_frontiers = measure.label(frontier_map, background=0)

        # Extract region properties
        regions = measure.regionprops(labelled_frontiers)

        candidate_points = []
        for region in regions:
            if region.area > 9:  # Filter out small noise frontiers
                centroid = region.centroid
                candidate_points.append([int(centroid[0]), int(centroid[1])])

        return candidate_points, labelled_frontiers

    def nearest_frontier(self, candidate_points):
        """
        Finds the nearest frontier to the robot's current position.
        This method calculates the distance from the robot's current position to each
        candidate frontier and returns a sorted list of frontiers in Cartesian coordinates.

        Args:
            candidate_points (list): List of frontier points in grid coordinates.

        Returns:
            numpy.ndarray: Array of frontier points sorted by distance in Cartesian coordinates.
        """

        # Calculate distances from the current pose to each candidate point
        distances = []
        for point in candidate_points:
            distance = self.pose_to_grid_distance(point, self.current_pose[:2])
            distances.append(distance)

        sorted_points = [point for _, point in sorted(zip(distances, candidate_points))]
        sorted_cartesian_points = [self.__map_to_position__(point) for point in sorted_points]

        return np.array(sorted_cartesian_points)

    def set_pose(self, pose):
        self.current_pose = pose

    def set_map(self, occupancy_map):
        self.occupancy_map = copy.deepcopy(occupancy_map)

    def pose_to_grid_distance(self, grid_point, pose_point):
        world_point = self.__map_to_position__(grid_point)
        return sqrt((pose_point[0] - world_point[0]) ** 2 + (pose_point[1] - world_point[1]) ** 2)

    def __map_to_position__(self, grid_point):
        x = grid_point[1] * self.map_resolution + self.map_origin[0]
        y = grid_point[0] * self.map_resolution + self.map_origin[1]
        return [x, y]
