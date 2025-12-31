import cv2
import numpy as np


class BoxKalmanFilter:
    """
    Simple Kalman Filter for tracking bounding box state (x, y, w, h).
    Used for smoothing and extrapolation.
    """
    def __init__(self, xyxy, class_idx, conf):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        # Initial state
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

        self.class_idx = class_idx
        self.conf = conf
        self.xyxy = xyxy

    def update(self, xyxy):
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        
        measurement = np.array([cx, cy, w, h], dtype=np.float32)
        self.kf.correct(measurement)
        self.kf.predict() 
        self.xyxy = xyxy

    def predict(self):
        prediction = self.kf.predict()
        cx, cy, w, h = prediction[:4].flatten()
        
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        
        return [x1, y1, x2, y2]