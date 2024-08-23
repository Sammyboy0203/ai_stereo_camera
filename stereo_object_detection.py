
# Import necessary libraries
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import logging
import time

class StereoObjectDetector:
    def __init__(self, use_stereo=False, detect_class='person', simulate_stereo=False):
        # Initialize detector parameters
        self.use_stereo = use_stereo
        self.simulate_stereo = simulate_stereo and use_stereo
        self.detect_class = detect_class
        self.model = YOLO('yolov8n.pt')
        self.camera = None
        self.logger = self._setup_logger()

        # Camera parameters
        self.frame_width = 2560  # 1280 * 2
        self.frame_height = 960
        self.bytes_per_pixel = 2  # oCam-1CGN-U uses 16-bit data
        self.focal_length = 500  # in pixels
        self.baseline = 0.1  # in meters (for stereo setup)
        self.max_init_attempts = 5
        self.init_delay = 5

        # Add stereo camera parameters
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.stereo_map_left = None
        self.stereo_map_right = None
        self.Q = None  # Add this line

        if use_stereo and not simulate_stereo:
            self.load_calibration_data()
            self.stereo_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16*16,  # Reduced from 16*32
                blockSize=7,  # Reduced from 11
                P1=8 * 3 * 7**2,
                P2=32 * 3 * 7**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

    def _setup_logger(self):
        # Set up logging for the detector
        logger = logging.getLogger('StereoObjectDetector')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_calibration_data(self):
        try:
            calibration_data = np.load('stereo_calibration.npy', allow_pickle=True).item()
            self.camera_matrix_left = calibration_data['mtx_left']
            self.camera_matrix_right = calibration_data['mtx_right']
            self.dist_coeffs_left = calibration_data['dist_left']
            self.dist_coeffs_right = calibration_data['dist_right']
            self.R = calibration_data['R']
            self.T = calibration_data['T']
            self.Q = calibration_data['Q']  # Ensure this line is present
            
            # Print calibration data for verification
            self.logger.debug(f"Camera matrix left: {self.camera_matrix_left}")
            self.logger.debug(f"Camera matrix right: {self.camera_matrix_right}")
            self.logger.debug(f"Distortion coeffs left: {self.dist_coeffs_left}")
            self.logger.debug(f"Distortion coeffs right: {self.dist_coeffs_right}")
            self.logger.debug(f"R: {self.R}")
            self.logger.debug(f"T: {self.T}")
            self.logger.debug(f"Q: {self.Q}")
            
            # Update focal length and baseline from calibration data
            self.focal_length = self.camera_matrix_left[0, 0]  # Assuming fx = fy
            self.baseline = abs(self.T[0])  # The baseline is the absolute X-component of T
            
            self.stereo_map_left = cv2.initUndistortRectifyMap(self.camera_matrix_left, self.dist_coeffs_left, self.R, self.Q[:3, :3], (self.frame_width, self.frame_height), cv2.CV_16SC2)
            self.stereo_map_right = cv2.initUndistortRectifyMap(self.camera_matrix_right, self.dist_coeffs_right, self.R, self.Q[:3, :3], (self.frame_width, self.frame_height), cv2.CV_16SC2)
            
            self.logger.debug(f"Stereo map left shape: {self.stereo_map_left[0].shape}, {self.stereo_map_left[1].shape}")
            self.logger.debug(f"Stereo map right shape: {self.stereo_map_right[0].shape}, {self.stereo_map_right[1].shape}")
            self.logger.debug(f"Frame width: {self.frame_width}, Frame height: {self.frame_height}")
            
            self.logger.info("Calibration data loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load calibration data: {e}")

    def start_camera(self):
        if self.use_stereo and not self.simulate_stereo:
            for attempt in range(self.max_init_attempts):
                self.logger.info(f"Attempting to initialize oCam-1CGN-U (attempt {attempt + 1}/{self.max_init_attempts})")
                try:
                    self.camera = cv2.VideoCapture(0)
                    
                    if not self.camera.isOpened():
                        self.logger.error("Failed to open camera")
                        time.sleep(self.init_delay)
                        continue

                    # Set camera properties
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    self.camera.set(cv2.CAP_PROP_BRIGHTNESS, -1)  # Increased brightness
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, -7)    # Adjusted exposure
                    self.camera.set(cv2.CAP_PROP_GAIN, 100)       # Increased gain
                    self.camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Disable automatic conversion

                    # Print camera properties
                    self.logger.info(f"Set width: {self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)}")
                    self.logger.info(f"Set height: {self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                    self.logger.info(f"Set brightness: {self.camera.get(cv2.CAP_PROP_BRIGHTNESS)}")
                    self.logger.info(f"Set exposure: {self.camera.get(cv2.CAP_PROP_EXPOSURE)}")
                    self.logger.info(f"Set gain: {self.camera.get(cv2.CAP_PROP_GAIN)}")

                    ret, frame = self.camera.read()
                    if not ret or frame is None:
                        self.logger.error("Failed to capture frame from oCam-1CGN-U")
                        self.camera.release()
                        time.sleep(self.init_delay)
                        continue

                    if frame.shape != (1, 2457600):
                        self.logger.error(f"Unexpected frame shape: {frame.shape}")
                        self.camera.release()
                        time.sleep(self.init_delay)
                        continue

                    self.logger.info("Successfully initialized oCam-1CGN-U")
                    return True
                except Exception as e:
                    self.logger.error(f"Error during camera setup: {str(e)}")
                    if self.camera is not None:
                        self.camera.release()
                    time.sleep(self.init_delay)

            self.logger.error("Failed to initialize camera after multiple attempts")
            return False

        elif self.simulate_stereo:
            self.camera = cv2.VideoCapture(0)
        else:
            self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            self.logger.error("Error: Could not open camera(s).")
            return False
        return True

    def process_frame(self, frame):
        self.logger.debug(f"Input frame shape: {frame.shape}")
        if frame.shape != (1, 2457600):
            self.logger.error(f"Unexpected frame shape: {frame.shape}")
            return None, None

        # Reshape the frame to 2D
        frame_2d = frame.reshape((960, 1280 * 2))
        
        # Split the frame into left and right images
        left = frame_2d[:, 1::2]
        right = frame_2d[:, ::2]
        
        # Convert from Bayer to BGR
        try:
            left_bgr = cv2.cvtColor(left, cv2.COLOR_BayerGB2BGR)
            right_bgr = cv2.cvtColor(right, cv2.COLOR_BayerGB2BGR)
        except Exception as e:
            self.logger.error(f"Error in Bayer to BGR conversion: {e}")
            return None, None

        return left_bgr, right_bgr

    def estimate_distance(self, bbox_left, bbox_right, left_frame, right_frame):
        try:
            self.logger.debug("Starting distance estimation")
            
            # Display original images
            cv2.imshow("Original Left", left_frame)
            cv2.imshow("Original Right", right_frame)
            
            # Try simple undistortion instead of full rectification
            left_rectified = cv2.undistort(left_frame, self.camera_matrix_left, self.dist_coeffs_left)
            right_rectified = cv2.undistort(right_frame, self.camera_matrix_right, self.dist_coeffs_right)
            
            self.logger.debug(f"Rectified image shapes: Left {left_rectified.shape}, Right {right_rectified.shape}")
            
            # Check if rectified images are all black
            if np.all(left_rectified == 0) or np.all(right_rectified == 0):
                self.logger.error("Rectified images are completely black")
                return None
            
            # Visualize rectified images
            cv2.imshow("Left Rectified", left_rectified)
            cv2.imshow("Right Rectified", right_rectified)
            
            # Convert rectified images to grayscale
            self.logger.debug("Converting to grayscale")
            gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
            
            # Downscale images
            scale_factor = 0.5
            gray_left_small = cv2.resize(gray_left, None, fx=scale_factor, fy=scale_factor)
            gray_right_small = cv2.resize(gray_right, None, fx=scale_factor, fy=scale_factor)
            
            # Compute disparity for the downscaled images
            self.logger.debug("Computing disparity")
            disparity = self.stereo_matcher.compute(gray_left_small, gray_right_small).astype(np.float32) / 16.0
            self.logger.debug(f"Disparity shape: {disparity.shape}")
            self.logger.debug(f"Disparity stats - Min: {np.min(disparity)}, Max: {np.max(disparity)}, Mean: {np.mean(disparity)}")
            
            # Upscale disparity to original size
            disparity = cv2.resize(disparity, (gray_left.shape[1], gray_left.shape[0]))
            
            # Get the disparity within the object's bounding box
            x1, y1, x2, y2 = bbox_left
            self.logger.debug(f"Bounding box coordinates: ({x1}, {y1}), ({x2}, {y2})")
            self.logger.debug(f"Image shape: {left_frame.shape}")
            object_disparity = disparity[y1:y2, x1:x2]
            
            # Filter out invalid disparities (expanded range)
            valid_disparities = object_disparity[(object_disparity > 0) & (object_disparity < 128)]
            
            if len(valid_disparities) == 0:
                self.logger.warning("No valid disparities found in the bounding box")
                return None
            
            # Use median disparity
            median_disparity = np.median(valid_disparities)
            
            self.logger.debug(f"Median disparity: {median_disparity}")
            
            if median_disparity <= 0:
                self.logger.warning("Invalid median disparity (<=0)")
                return None
            
            # Calculate distance using the disparity formula
            distance = (self.focal_length * self.baseline) / median_disparity
            
            self.logger.debug(f"Calculated distance: {distance}")
            
            if not np.isfinite(distance):
                self.logger.warning("Computed distance is not finite")
                return None
            
            # Visualize disparity map
            normalized_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow("Disparity Map", normalized_disparity)
            
            return round(distance, 2)
        except Exception as e:
            self.logger.error(f"Error computing distance: {e}")
            return None

    def detect_objects(self, left_frame, right_frame):
        # Implement object detection using YOLOv8 with lower confidence threshold
        results_left = self.model(left_frame, conf=0.3)
        results_right = self.model(right_frame, conf=0.3)
        
        detections = []
        
        for r_left, r_right in zip(results_left, results_right):
            boxes_left = r_left.boxes
            boxes_right = r_right.boxes
            
            # Create a list of tuples (box_left, box_right) where both exist
            matched_boxes = []
            for box_left in boxes_left:
                cls_left = self.model.names[int(box_left.cls[0])]
                if cls_left == self.detect_class:
                    # Find a matching box in the right frame
                    for box_right in boxes_right:
                        cls_right = self.model.names[int(box_right.cls[0])]
                        if cls_right == self.detect_class:
                            matched_boxes.append((box_left, box_right))
                            break
            
            for box_left, box_right in matched_boxes:
                confidence_left = round(float(box_left.conf[0]), 2)
                confidence_right = round(float(box_right.conf[0]), 2)
                
                bbox_left = list(map(int, box_left.xyxy[0]))
                bbox_right = list(map(int, box_right.xyxy[0]))
                
                distance = self.estimate_distance(bbox_left, bbox_right, left_frame, right_frame)
                
                detections.append({
                    'class': self.detect_class,
                    'confidence': (confidence_left + confidence_right) / 2,
                    'bbox_left': bbox_left,
                    'bbox_right': bbox_right,
                    'distance': distance
                })
        
        return detections

    def adjust_camera_settings(self, brightness=None, exposure=None, gain=None):
        if self.camera is not None:
            if brightness is not None:
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            if exposure is not None:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            if gain is not None:
                self.camera.set(cv2.CAP_PROP_GAIN, gain)
            
            self.logger.info(f"Updated camera settings:")
            self.logger.info(f"Brightness: {self.camera.get(cv2.CAP_PROP_BRIGHTNESS)}")
            self.logger.info(f"Exposure: {self.camera.get(cv2.CAP_PROP_EXPOSURE)}")
            self.logger.info(f"Gain: {self.camera.get(cv2.CAP_PROP_GAIN)}")

    def run(self):
        if not self.start_camera():
            self.logger.error("Failed to start camera. Exiting.")
            return

        self.logger.info(f"Starting detection. Press 'q' to quit, 'c' to change class, 'b' to adjust brightness, 'e' to adjust exposure, 'g' to adjust gain.")
        
        while True:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.error("Failed to grab frame")
                    break

                if self.use_stereo and not self.simulate_stereo:
                    left_bgr, right_bgr = self.process_frame(frame)
                    if left_bgr is None or right_bgr is None:
                        self.logger.warning("Failed to process frame")
                        continue
                    
                    detections = self.detect_objects(left_bgr, right_bgr)
                    
                    for detection in detections:
                        cls = detection['class']
                        confidence = detection['confidence']
                        bbox_left = detection['bbox_left']
                        bbox_right = detection['bbox_right']
                        distance = detection['distance']
                        
                        # Draw bounding boxes and information on both frames
                        for frame, bbox in [(left_bgr, bbox_left), (right_bgr, bbox_right)]:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{cls} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            if distance is not None:
                                cv2.putText(frame, f"Dist: {distance}m", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, "Dist: N/A", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        self.logger.info(f"Detected {cls} with confidence {confidence:.2f} at distance {distance}m")
                    
                    cv2.imshow("Left Camera", left_bgr)
                    cv2.imshow("Right Camera", right_bgr)
                else:
                    # Handle non-stereo cases
                    processed_frame = self.detect_objects(frame)
                    cv2.imshow("Camera", processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.detect_class = input("Enter new class to detect: ")
                    self.logger.info(f"Now detecting: {self.detect_class}")
                elif key == ord('b'):
                    brightness = float(input("Enter new brightness value (-64 to 64): "))
                    self.adjust_camera_settings(brightness=brightness)
                elif key == ord('e'):
                    exposure = float(input("Enter new exposure value (-13 to -1): "))
                    self.adjust_camera_settings(exposure=exposure)
                elif key == ord('g'):
                    gain = float(input("Enter new gain value (0 to 240): "))
                    self.adjust_camera_settings(gain=gain)

            except Exception as e:
                self.logger.error(f"An error occurred: {str(e)}")
                break

        self.camera.release()
        cv2.destroyAllWindows()

def main():
    # Parse command-line arguments and run the detector
    parser = argparse.ArgumentParser(description="Stereo Object Detection Demo")
    parser.add_argument("--stereo", action="store_true", help="Use stereo camera mode")
    parser.add_argument("--simulate", action="store_true", help="Simulate stereo camera")
    parser.add_argument("--detect_class", type=str, default="person", help="Class to detect")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    detector = StereoObjectDetector(use_stereo=args.stereo, detect_class=args.detect_class, simulate_stereo=args.simulate)
    detector.run()

if __name__ == "__main__":
    main()