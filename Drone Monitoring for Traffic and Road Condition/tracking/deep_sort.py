# deep_sort.py

import numpy as np
from collections import OrderedDict
import cv2

class SimpleTracker:
    def __init__(self, max_lost=30):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.lost = OrderedDict()
        self.max_lost = max_lost

    def add_object(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.lost[self.next_object_id] = 0
        self.next_object_id += 1

    def remove_object(self, object_id):
        del self.objects[object_id]
        del self.lost[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.lost.keys()):
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    self.remove_object(object_id)
            return self.objects

        new_objects = {}
        input_centroids = np.array([self._centroid(box) for box in detections])

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.add_object(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = self._distance(object_centroids, input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.lost[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    self.remove_object(object_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.add_object(input_centroids[col])

        return self.objects

    def _centroid(self, box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _distance(self, centroidsA, centroidsB):
        A = np.array(centroidsA)
        B = np.array(centroidsB)
        return np.linalg.norm(A[:, np.newaxis] - B, axis=2)
