#!/usr/bin/env python3
"""
Sample ML Processor for Mentra Real-Time Video Stream

This example demonstrates how to:
1. Connect to the HLS stream from Mentra glasses
2. Extract frames for processing
3. Apply ML models (object detection, OCR, etc.)
4. Display or save results

Requirements:
    pip install opencv-python requests m3u8 numpy pillow
"""

import cv2
import time
import requests
import m3u8
import numpy as np
from urllib.parse import urljoin
from datetime import datetime
import json
import os
from typing import Optional, Tuple, List
import threading
import queue

class MentraStreamProcessor:
    """Process video stream from Mentra glasses"""
    
    def __init__(self, server_url: str = "http://localhost:3000"):
        self.server_url = server_url
        self.hls_url: Optional[str] = None
        self.is_processing = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.processed_frames = 0
        
    def get_stream_status(self) -> dict:
        """Get current stream status from the server"""
        try:
            response = requests.get(f"{self.server_url}/api/stream/status")
            return response.json()
        except Exception as e:
            print(f"Error getting stream status: {e}")
            return {}
    
    def wait_for_stream(self, timeout: int = 60) -> bool:
        """Wait for stream to become active"""
        print("Waiting for stream to start...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_stream_status()
            
            if status.get('status') == 'active' and status.get('hlsUrl'):
                self.hls_url = status['hlsUrl']
                print(f"Stream active! HLS URL: {self.hls_url}")
                return True
            
            time.sleep(2)
        
        print("Timeout waiting for stream")
        return False
    
    def extract_frames_from_hls(self):
        """Extract frames from HLS stream"""
        if not self.hls_url:
            print("No HLS URL available")
            return
        
        print(f"Connecting to HLS stream: {self.hls_url}")
        
        # For direct OpenCV capture (works with some HLS streams)
        cap = cv2.VideoCapture(self.hls_url)
        
        if not cap.isOpened():
            print("Failed to open HLS stream with OpenCV")
            return
        
        print("Successfully connected to stream")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = int(fps / 5)  # Process 5 frames per second
        frame_count = 0
        
        while self.is_processing:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame, retrying...")
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # Only process every Nth frame to reduce load
            if frame_count % frame_interval == 0:
                # Add frame to queue for processing
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                    self.processed_frames += 1
        
        cap.release()
        print("Frame extraction stopped")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame with ML models
        
        This is where you would add your ML processing:
        - Object detection (YOLO, etc.)
        - Face recognition
        - OCR (Tesseract, EasyOCR)
        - Custom models
        """
        height, width = frame.shape[:2]
        
        # Example: Simple edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        # Example: Color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])
        
        # Example: Motion detection (would need previous frame)
        # You can implement frame differencing here
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': self.processed_frames,
            'dimensions': {'width': width, 'height': height},
            'analysis': {
                'edge_density': edge_pixels / (width * height),
                'avg_hue': float(avg_hue),
                'avg_saturation': float(avg_saturation),
                'avg_brightness': float(avg_value),
            }
        }
        
        # Add your ML model predictions here
        # Example with a hypothetical object detection model:
        # detections = self.detect_objects(frame)
        # result['objects'] = detections
        
        return result
    
    def processing_worker(self):
        """Worker thread for processing frames"""
        while self.is_processing or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)
                result = self.process_frame(frame)
                
                # Print or save results
                print(f"Frame {result['frame_number']}: "
                      f"Edge density: {result['analysis']['edge_density']:.3f}, "
                      f"Brightness: {result['analysis']['avg_brightness']:.1f}")
                
                # Optionally display the frame
                # cv2.imshow('Mentra Stream', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     self.stop()
                
                # Save frame periodically
                if self.processed_frames % 30 == 0:  # Every 30 frames
                    filename = f"frame_{self.processed_frames:06d}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved {filename}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
    
    def start(self):
        """Start processing the stream"""
        if not self.wait_for_stream():
            return False
        
        self.is_processing = True
        
        # Start frame extraction thread
        extraction_thread = threading.Thread(target=self.extract_frames_from_hls)
        extraction_thread.start()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.processing_worker)
        processing_thread.start()
        
        print("Processing started. Press Ctrl+C to stop.")
        
        try:
            while self.is_processing:
                time.sleep(1)
                if self.processed_frames > 0 and self.processed_frames % 100 == 0:
                    print(f"Processed {self.processed_frames} frames")
        except KeyboardInterrupt:
            print("\nStopping processing...")
            self.stop()
        
        extraction_thread.join()
        processing_thread.join()
        
        return True
    
    def stop(self):
        """Stop processing"""
        self.is_processing = False
        cv2.destroyAllWindows()
        print(f"Processing stopped. Total frames processed: {self.processed_frames}")


class ObjectDetectionProcessor(MentraStreamProcessor):
    """
    Extended processor with object detection capabilities
    
    To use this, install:
        pip install ultralytics  # For YOLO
    """
    
    def __init__(self, server_url: str = "http://localhost:3000"):
        super().__init__(server_url)
        self.model = None
        # Uncomment to use YOLO:
        # from ultralytics import YOLO
        # self.model = YOLO('yolov8n.pt')  # Use yolov8n for speed
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process frame with object detection"""
        result = super().process_frame(frame)
        
        if self.model:
            # Run YOLO detection
            # detections = self.model(frame)
            # objects = []
            # for r in detections:
            #     for box in r.boxes:
            #         objects.append({
            #             'class': r.names[int(box.cls)],
            #             'confidence': float(box.conf),
            #             'bbox': box.xyxy.tolist()[0]
            #         })
            # result['objects'] = objects
            pass
        
        return result


class OCRProcessor(MentraStreamProcessor):
    """
    Extended processor with OCR capabilities
    
    To use this, install:
        pip install easyocr
    """
    
    def __init__(self, server_url: str = "http://localhost:3000"):
        super().__init__(server_url)
        self.reader = None
        # Uncomment to use EasyOCR:
        # import easyocr
        # self.reader = easyocr.Reader(['en'])
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process frame with OCR"""
        result = super().process_frame(frame)
        
        if self.reader:
            # Run OCR
            # text_results = self.reader.readtext(frame)
            # texts = []
            # for (bbox, text, prob) in text_results:
            #     if prob > 0.5:  # Confidence threshold
            #         texts.append({
            #             'text': text,
            #             'confidence': float(prob),
            #             'bbox': bbox
            #         })
            # result['text'] = texts
            pass
        
        return result


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Mentra video stream')
    parser.add_argument('--server', default='http://localhost:3000',
                       help='Mentra server URL')
    parser.add_argument('--processor', default='basic',
                       choices=['basic', 'object', 'ocr'],
                       help='Processor type to use')
    
    args = parser.parse_args()
    
    # Select processor
    if args.processor == 'object':
        processor = ObjectDetectionProcessor(args.server)
    elif args.processor == 'ocr':
        processor = OCRProcessor(args.server)
    else:
        processor = MentraStreamProcessor(args.server)
    
    print(f"Starting {args.processor} processor...")
    print(f"Server: {args.server}")
    print("Make sure the Mentra stream is running!")
    
    processor.start()


if __name__ == "__main__":
    main()