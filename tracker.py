from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2

class Tracker:
    def __init__(self, embedder="mobilenet", use_cuda=True):
        """
        Initialize DeepSort tracker with optimized parameters and specified embedder model
        
        Args:
            embedder (str): Re-identification model to use. Options include: 
                           'mobilenet' (default, good balance of speed/accuracy),
                           'efficientnet' (more accurate but slower),
                           'osnet_ain' (designed specifically for person re-ID)
            use_cuda (bool): Whether to use GPU acceleration for the embedder model
        """
        # Initialize DeepSort tracker with optimized parameters
        self.object_tracker = DeepSort(
            max_age=30,              # Maximum frames to keep lost tracks (increased for better re-ID)
            n_init=3,                # Frames needed to initialize a track (increased for reliability)
            nms_max_overlap=0.3,     # Non-maximum suppression threshold  
            max_cosine_distance=0.7, # Cosine distance threshold for appearance features (reduced for stricter matching)
            nn_budget=100,           # Maximum size for appearance descriptors gallery (increased for better history)
            override_track_class=None,  # Use default track class with built-in Kalman filter
            embedder=embedder,       # Feature extractor model for re-ID
            half=not use_cuda,       # Use half precision only if not using CUDA
            bgr=True,                # Use BGR format for processing
            embedder_model_name=None,  # Default embedder model
            embedder_wts=None,       # Default embedder weights
            polygon=False,           # Use rectangle, not polygons
            today=None
        )
        
        # Dictionary to store track history for smoother visualization
        self.track_history = {}
        self.max_history = 5
        
        # Store appearance features for each track (optional for advanced re-ID)
        self.appearance_features = {}
        
        # Track quality metrics (optional)
        self.track_quality = {}
        
    def track(self, detections, frame):
        """
        Update tracks using the provided detections and the internal Kalman filter
        
        Args:
            detections: List of detections in format [(bbox, confidence, class_id), ...]
            frame: The current video frame
            
        Returns:
            tracking_ids: List of track IDs
            boxes: List of bounding boxes
            velocities: List of velocity estimates from Kalman filter
        """
        # Update tracks using DeepSort's built-in Kalman filter and re-ID
        tracks = self.object_tracker.update_tracks(detections, frame=frame)
        
        tracking_ids = []
        boxes = []
        velocities = []  # Store Kalman filter's velocity estimates
        confidences = []  # Track confidence based on association quality
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            tracking_ids.append(track_id)
            
            # Get bounding box from DeepSort's Kalman filter state
            ltrb = track.to_ltrb()
            boxes.append(ltrb)
            
            # Access DeepSort's internal KalmanBoxTracker to get velocity estimates (optional)
            if hasattr(track, 'kalman_filter') and track.kalman_filter:
                # Extract velocity from state [x, y, a, h, vx, vy, va, vh]
                try:
                    vx = track.kalman_filter.x[4]
                    vy = track.kalman_filter.x[5]
                    velocities.append((vx, vy))
                except (AttributeError, IndexError):
                    velocities.append((0, 0))  # Default if not available
            else:
                velocities.append((0, 0))  # Default if Kalman filter not accessible
            
            # Store appearance feature from DeepSort (if available)
            if hasattr(track, 'features') and len(track.features) > 0:
                self.appearance_features[track_id] = track.features[-1]  # Latest feature
            
            # Track quality metric based on time since update
            if hasattr(track, 'time_since_update'):
                confidence = 1.0 / (1.0 + track.time_since_update)
                confidences.append(confidence)
            else:
                confidences.append(1.0)  # Default confidence
                
            # Track bounding box history for smoother visualization (optional)
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                
            self.track_history[track_id].append(ltrb)
            if len(self.track_history[track_id]) > self.max_history:
                self.track_history[track_id].pop(0)
        
        # Clean up history for disappeared tracks (optional)
        current_track_ids = set(tracking_ids)
        self.track_history = {k: v for k, v in self.track_history.items() if k in current_track_ids}
        
        return tracking_ids, boxes, velocities, confidences
    
    def get_smoothed_boxes(self, track_id):
        """Get smoothed bounding boxes using track history (optional)"""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 3:
            return None
            
        # Use median filtering for smoother bounding boxes
        history = np.array(self.track_history[track_id])
        
        # Compute center points and dimensions
        centers_x = (history[:, 0] + history[:, 2]) / 2
        centers_y = (history[:, 1] + history[:, 3]) / 2
        widths = history[:, 2] - history[:, 0]
        heights = history[:, 3] - history[:, 1]
        
        # Apply median filter
        smooth_center_x = np.median(centers_x)
        smooth_center_y = np.median(centers_y)
        smooth_width = np.median(widths)
        smooth_height = np.median(heights)
        
        # Reconstruct smoothed box
        smooth_x1 = smooth_center_x - smooth_width/2
        smooth_y1 = smooth_center_y - smooth_height/2
        smooth_x2 = smooth_center_x + smooth_width/2
        smooth_y2 = smooth_center_y + smooth_height/2
        
        return [smooth_x1, smooth_y1, smooth_x2, smooth_y2]
    
    def get_feature_similarity(self, track_id1, track_id2):
        """
        Calculate similarity between two tracks based on appearance features (optional)
        Useful for advanced re-ID applications
        """
        if track_id1 not in self.appearance_features or track_id2 not in self.appearance_features:
            return 0.0
            
        feature1 = self.appearance_features[track_id1]
        feature2 = self.appearance_features[track_id2]
        
        # Normalize features
        feature1 = feature1 / np.linalg.norm(feature1)
        feature2 = feature2 / np.linalg.norm(feature2)
        
        # Calculate cosine similarity
        similarity = np.dot(feature1, feature2)
        return similarity
