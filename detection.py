import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from filterpy.kalman import ExtendedKalmanFilter

# Custom Extended Kalman Filter optimized for human tracking
class HumanEKF:
    def __init__(self, dim_x=7, dim_z=4):
        self.kf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # State transition matrix - constant velocity model
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x
            [0, 1, 0, 0, 0, 1, 0],  # y
            [0, 0, 1, 0, 0, 0, 1],  # s (scale)
            [0, 0, 0, 1, 0, 0, 0],  # r (aspect ratio)
            [0, 0, 0, 0, 1, 0, 0],  # x'
            [0, 0, 0, 0, 0, 1, 0],  # y'
            [0, 0, 0, 0, 0, 0, 1],  # s'
        ])
        
        # Initialize state covariance matrix - tuned for human movement
        self.kf.P *= 10
        self.kf.P[4:, 4:] *= 100  # Give velocity components higher uncertainty
        
        # Measurement noise - tuned for human detection
        self.kf.R = np.diag([5, 5, 20, 2])  # Lower for position, higher for area, lower for aspect ratio
        
        # Process noise - optimized for better bounding box size adaptation
        self.kf.Q = np.eye(dim_x) * 0.03  # Base process noise
        self.kf.Q[0:2, 0:2] *= 0.5  # Position variables (humans move but not erratically)
        self.kf.Q[2:4, 2:4] *= 0.5  # Size variables - increased from 0.2 for better size adaptation
        self.kf.Q[4:6, 4:6] *= 0.8  # Velocity variables for position (can change moderately)
        self.kf.Q[6:, 6:] *= 0.5    # Velocity variable for size - increased from 0.3 for faster size changes

    def predict(self):
        self.kf.predict()
        return self.kf.x
        
    def update(self, measurement):
        # Reshape measurement to be a column vector (4,1) instead of (4,)
        measurement = np.array(measurement).reshape(-1, 1)
        self.kf.update(measurement, HJacobian=self.HJacobian, Hx=self.Hx)
        return self.kf.x
    
    def HJacobian(self, x):
        # Jacobian of the measurement function
        return np.array([
            [1, 0, 0, 0, 0, 0, 0],  # x measurement
            [0, 1, 0, 0, 0, 0, 0],  # y measurement
            [0, 0, 1, 0, 0, 0, 0],  # s measurement
            [0, 0, 0, 1, 0, 0, 0],  # r measurement
        ])
    
    def Hx(self, x):
        # Measurement function - linear for this implementation
        return np.array([x[0], x[1], x[2], x[3]]).reshape(-1, 1)  # Return as column vector

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a larger model for better accuracy with humans if needed

# Initialize DeepSORT with parameters optimized for human tracking
tracker = DeepSort(
    max_age=45,         # Humans might be occluded for longer periods
    nn_budget=150,      # Keep more appearance features for humans
    embedder="mobilenet"
)

# Dictionary to store our custom EKFs for each human track
human_trackers = {}

# Open webcam
cap = cv2.VideoCapture(1)

# Font settings for cleaner display
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

# Confidence threshold
confidence_threshold = 0.5  # Set your desired confidence threshold here

# Variable to store the selected human track ID
selected_track_id = None

# Bounding box scaling factor to match DeepSORT size
bbox_scale_factor = 1.1  # Adjust if needed

# Function to handle mouse clicks
def select_human(event, x, y, flags, param):
    global selected_track_id
    if event == cv2.EVENT_LBUTTONDOWN:
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_track_id = track.track_id
                print(f"Selected Human #{selected_track_id}")
                break

# Set mouse callback
cv2.namedWindow("Human Tracking with YOLOv8 + DeepSORT + EKF")
cv2.setMouseCallback("Human Tracking with YOLOv8 + DeepSORT + EKF", select_human)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 inference - ONLY detect humans (class 0)
    results = model(frame, classes=[0])  # Explicitly filter for class 0 (person)
    detections = []
    
    for result in results:
        for box in result.boxes:
            # We don't need to check class here since we filtered during inference
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            
            # Filter detections based on confidence threshold
            if confidence >= confidence_threshold:
                # Using class 0 for all detections since we filtered for persons
                detections.append(([x1, y1, x2, y2], confidence, 0))
    
    # Update DeepSORT tracker with filtered human detections
    tracked_objects = tracker.update_tracks(detections, frame=frame)
    
    # Draw a title showing how many humans are being tracked
    num_tracked = sum(1 for track in tracked_objects if track.is_confirmed())
    cv2.putText(frame, f'Tracking {num_tracked} humans', (10, 30), 
                font, 1, (0, 128, 255), 2)
    
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
            
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        
        # If a human is selected, only track that human
        if selected_track_id is not None and track_id != selected_track_id:
            continue
        
        # Calculate center, width, height, area, and aspect ratio
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w/2
        center_y = y1 + h/2
        s = w * h  # area
        r = w / float(h) if h != 0 else 1.0  # aspect ratio, avoid div by zero
        
        measurement = np.array([center_x, center_y, s, r])
        
        # Initialize or update our custom EKF for this human track
        if track_id not in human_trackers:
            human_trackers[track_id] = HumanEKF()
            # Initialize with current detection
            human_trackers[track_id].kf.x[:4] = measurement.reshape(-1, 1)
        else:
            # Predict new position
            human_trackers[track_id].predict()
            
            # Update with current measurement
            improved_state = human_trackers[track_id].update(measurement)
            improved_state = improved_state.flatten()  # Convert back to 1D array
            
            # Use the improved state to refine bounding box
            improved_center_x, improved_center_y = improved_state[0], improved_state[1]
            improved_s = improved_state[2]  # area
            improved_r = improved_state[3]  # aspect ratio
            
            # Calculate width and height from area and aspect ratio with scaling factor
            improved_w = int(np.sqrt(improved_s * improved_r) * bbox_scale_factor)
            improved_h = int(np.sqrt(improved_s / improved_r) * bbox_scale_factor) if improved_r != 0 else int(np.sqrt(improved_s) * bbox_scale_factor)
            
            # Ensure width and height aren't too small
            improved_w = max(improved_w, int(w * 0.9))  # At least 90% of original width
            improved_h = max(improved_h, int(h * 0.9))  # At least 90% of original height
            
            improved_x1 = int(improved_center_x - improved_w/2)
            improved_y1 = int(improved_center_y - improved_h/2)
            improved_x2 = int(improved_center_x + improved_w/2)
            improved_y2 = int(improved_center_y + improved_h/2)
            
            # Draw improved bounding box (green)
            cv2.rectangle(frame, (improved_x1, improved_y1), (improved_x2, improved_y2), (0, 255, 0), 2)
        
        # Draw original DeepSORT bounding box (black)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, f'Human #{track_id}', (x1, y1 - 5),
                   font, font_scale, (0, 0, 0), font_thickness)
        
        # Display confidence score from original detection
        confidence = next((conf for bbox, conf, _ in detections 
                          if x1 <= bbox[0] and y1 <= bbox[1] 
                          and x2 >= bbox[2] and y2 >= bbox[3]), None)
        if confidence:
            cv2.putText(frame, f'conf:{confidence:.2f}', (x1, y2 + 15), 
                       font, font_scale, (0, 0, 0), font_thickness)
    
    # Clean up trackers for disappeared humans
    current_track_ids = set(track.track_id for track in tracked_objects if track.is_confirmed())
    human_trackers = {k: v for k, v in human_trackers.items() if k in current_track_ids}
    
    # Display output
    cv2.imshow("Human Tracking with YOLOv8 + DeepSORT + EKF", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
