import numpy as np
from collections import OrderedDict
import cv2

# A simple centroid-based tracking class
class SimpleTracker:
    def __init__(self, max_lost=30):
        # ID to assign to the next new object
        self.next_object_id = 0
        # Dictionary that holds current tracked object centroids
        self.objects = OrderedDict()
        # Dictionary that tracks how long an object has been missing
        self.lost = OrderedDict()
        # Maximum number of frames an object can be lost before being removed
        self.max_lost = max_lost

    # Register a new object with its centroid
    def add_object(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.lost[self.next_object_id] = 0
        self.next_object_id += 1

    # Remove an object by its ID
    def remove_object(self, object_id):
        del self.objects[object_id]
        del self.lost[object_id]

    # Update tracker with new detection bounding boxes
    def update(self, detections):
        # Case 1: No detections received, increase lost count
        if len(detections) == 0:
            for object_id in list(self.lost.keys()):
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    self.remove_object(object_id)
            return self.objects

        # Convert detections to centroids
        input_centroids = np.array([self._centroid(box) for box in detections])

        # Case 2: No currently tracked objects, register all new detections
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.add_object(centroid)
        else:
            # Get current object IDs and their centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance matrix between current objects and new detections
            D = self._distance(object_centroids, input_centroids)

            # Match the closest pairs (lower distance = better match)
            rows = D.min(axis=1).argsort()     # Sort by closest match row-wise
            cols = D.argmin(axis=1)[rows]      # Find best matching column for each row

            used_rows = set()
            used_cols = set()

            # Assign matched centroids
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]  # Update centroid
                self.lost[object_id] = 0                        # Reset lost count
                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects (potentially lost)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    self.remove_object(object_id)

            # Handle new detections that do not match existing objects
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.add_object(input_centroids[col])

        return self.objects

    # Convert a bounding box to a centroid (center point)
    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    # Compute pairwise Euclidean distance between two sets of centroids
    def _distance(self, centroidsA, centroidsB):
        A = np.array(centroidsA)
        B = np.array(centroidsB)
        return np.linalg.norm(A[:, np.newaxis] - B, axis=2)
