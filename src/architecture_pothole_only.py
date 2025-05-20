from ultralytics import YOLO
import cv2
import math
from collections import defaultdict
import numpy as np
import torch
from core.settings import settings
from starlette.concurrency import run_in_threadpool
import datetime

class YOLOModel:
    
    def __init__(self):
        # Only initialize pothole model
        self.pothole_model = YOLO(settings.pothole_detection_weight)
        self.pothole_class_names = ['Pothole']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.warmup_pothole()
        self.pothole_track_history = defaultdict(lambda: [])

    def warmup_pothole(self):
        if self.pothole_model:
            print("Warming up pothole detection model")
            try:   
                frame = cv2.imread("test.jpg")
                _ = self.pothole_model(frame)
                print("Pothole Detection Model warmed up")
            except Exception as e:
                print(f"Error during warmup: {str(e)}")
                exit()
        else:
            print("Pothole Model is not available. Skipping warmup.")

    async def detect_and_track_potholes_and_draw(self, image):
        """
        Detect and track potholes in the input image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple: (detections, annotated_image)
                - detections: List of tuples (track_id, bounding_box, confidence)
                - annotated_image: Image with pothole detection annotations
        """
        if image.size == 0:
            print("No image to detect potholes on.")
            return False, image
            
        # Run detection with tracking
        results = self.pothole_model.track(image, persist=True, device=self.device)
        
        # Check if results are valid
        if not results or len(results) == 0 or results[0] is None or results[0].boxes is None:
            print("No pothole detection results")
            return False, image
            
        boxes = results[0].boxes.xywh.cpu()
        
        # Check if any detections
        if boxes.size(0) == 0:
            return False, image
            
        try:
            # Extract tracking IDs, confidence scores
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [i for i in range(len(boxes))]
            confidence = results[0].boxes.conf.cpu().tolist()
            detections = []
            
            # Get annotated image with default plotting
            annotated_image = results[0].plot()
            
            # Process each detected pothole
            for box, track_id, conf in zip(boxes, track_ids, confidence):
                if conf < settings.pothole_detection_thresh:
                    continue
                    
                # Get coordinates
                x, y, w, h = box
                
                # Update tracking history
                track = self.pothole_track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:  # Keep history limited
                    track.pop(0)
                    
                # Create bounding box info
                bounding_box = [int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)]
                detections.append((track_id, bounding_box, conf))
                
                # Draw tracking lines
                if len(track) > 1:
                    points = np.hstack(track).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(annotated_image, [points], isClosed=False, color=(0, 0, 255), thickness=2)
                
                # Add text label showing detection ID and confidence
                label = f"Pothole #{track_id} ({conf:.2f})"
                cv2.putText(annotated_image, label, (int(x), int(y - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            return detections, annotated_image
            
        except Exception as e:
            print(f"Error in pothole detection: {str(e)}")
            return False, image

    async def pipeline_pothole_and_draw(self, image):
        """
        Run the pothole detection pipeline and draw results.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple: (results, annotated_image)
                - results: List of pothole detections or False if none
                - annotated_image: Image with pothole annotations
        """
        # Detect and track potholes
        pothole_results, annotated_image = await self.detect_and_track_potholes_and_draw(image)
        
        if not pothole_results:
            return False, image
            
        # Add timestamp and count of potholes
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_image, f"Time: {timestamp}", (10, 25), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_image, f"Potholes: {len(pothole_results)}", (10, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return pothole_results, annotated_image
