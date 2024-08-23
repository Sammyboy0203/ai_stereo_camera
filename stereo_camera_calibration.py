import cv2
import numpy as np
import glob
import os
import time

class StereoCalibrator:
    def __init__(self):
        self.chessboard_size = (9, 6)  # 10x7 squares means 9x6 internal corners
        self.square_size = 0.03  # 3cm per square (21cm / 7 squares)
        self.calibration_folder = 'calibration_images'
        self.camera = None
        self.max_init_attempts = 5
        self.init_delay = 5
        self.max_capture_attempts = 100
        self.frame_width = 1280
        self.frame_height = 960
        self.camera_model = None
        self.frame_count = 0
        self.bytes_per_pixel = 2  # oCam-1CGN-U uses 16-bit data

    def setup_camera(self, brightness=32, exposure=-4, gain=64):
        for attempt in range(self.max_init_attempts):
            print(f"Attempting to initialize oCam-1CGN-U (attempt {attempt + 1}/{self.max_init_attempts})")
            try:
                self.camera = cv2.VideoCapture(0)
                
                if not self.camera.isOpened():
                    print("Failed to open camera")
                    time.sleep(self.init_delay)
                    continue

                # Set camera properties
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # 1280 * 2
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
                self.camera.set(cv2.CAP_PROP_GAIN, gain)
                self.camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Disable automatic conversion

                # Print camera properties
                print(f"Set width: {self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)}")
                print(f"Set height: {self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                print(f"Set brightness: {self.camera.get(cv2.CAP_PROP_BRIGHTNESS)}")
                print(f"Set exposure: {self.camera.get(cv2.CAP_PROP_EXPOSURE)}")
                print(f"Set gain: {self.camera.get(cv2.CAP_PROP_GAIN)}")

                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("Failed to capture frame from oCam-1CGN-U")
                    self.camera.release()
                    time.sleep(self.init_delay)
                    continue

                self.frame_size = frame.size
                self.frame_shape = frame.shape
                print(f"Captured frame shape: {self.frame_shape}")
                print(f"Captured frame size: {self.frame_size}")
                print(f"Frame data type: {frame.dtype}")
                print(f"Frame min value: {frame.min()}, max value: {frame.max()}")
                print("Successfully initialized oCam-1CGN-U")
                
                self.detect_camera_model()
                return True
            except Exception as e:
                print(f"Error during camera setup: {str(e)}")
                if self.camera is not None:
                    self.camera.release()
                time.sleep(self.init_delay)

        print(f"Failed to initialize camera after {self.max_init_attempts} attempts")
        return False

    def detect_camera_model(self):
        self.camera_model = "oCam-1CGN-U"
        print(f"Detected camera model: {self.camera_model}")

    def process_frame(self, frame):
        if frame.shape != (1, 2457600):
            print(f"Unexpected frame shape: {frame.shape}")
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
            print(f"Error in Bayer to BGR conversion: {e}")
            return None, None

        return left_bgr, right_bgr

    def split_frame(self, frame):
        # This method is no longer needed as we split the frame in process_frame
        # But we'll keep it for compatibility
        height, width = frame.shape[:2]
        mid = width // 2
        left_frame = frame[:, :mid]
        right_frame = frame[:, mid:]
        return left_frame, right_frame

    def capture_calibration_images(self, num_images=20):
        if not os.path.exists(self.calibration_folder):
            os.makedirs(self.calibration_folder)

        print("Preparing to capture calibration images...")
        print("Please move the chessboard around in the camera's view.")
        print("Press 'c' to capture an image, 'q' to finish capturing.")

        count = 0
        while count < num_images:
            ret, frame = self.camera.read()

            if not ret or frame is None:
                print("Failed to capture frame")
                continue

            left_bgr, right_bgr = self.process_frame(frame)
            if left_bgr is None or right_bgr is None:
                print("Failed to process frame")
                continue

            # Display processed frames
            cv2.imshow("Left Camera", left_bgr)
            cv2.imshow("Right Camera", right_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                filename_left = os.path.join(self.calibration_folder, f'left_{count:02d}.jpg')
                filename_right = os.path.join(self.calibration_folder, f'right_{count:02d}.jpg')
                cv2.imwrite(filename_left, left_bgr)
                cv2.imwrite(filename_right, right_bgr)
                print(f"Captured image pair {count + 1}/{num_images}")
                count += 1
            elif key == ord('q'):
                print("Capture process stopped by user.")
                break

        cv2.destroyAllWindows()
        print(f"Captured {count} image pairs for calibration.")
        return count > 0

    def calibrate_cameras(self):
        # Prepare object points
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        # Arrays to store object points and image points from all images
        objpoints = []  # 3d points in real world space
        imgpoints_left = []  # 2d points in left image plane
        imgpoints_right = []  # 2d points in right image plane

        # Get list of calibration images
        images_left = sorted(glob.glob(os.path.join(self.calibration_folder, 'left_*.jpg')))
        images_right = sorted(glob.glob(os.path.join(self.calibration_folder, 'right_*.jpg')))

        for left_img, right_img in zip(images_left, images_right):
            img_left = cv2.imread(left_img)
            img_right = cv2.imread(right_img)
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size, None)

            if ret_left and ret_right:
                objpoints.append(objp)
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)

        # Calibrate each camera individually
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

        # Stereo calibration
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx_left, dist_left,
            mtx_right, dist_right,
            gray_left.shape[::-1], criteria=criteria, flags=flags)

        # Stereo rectification
        rect_left, rect_right, proj_mat_left, proj_mat_right, Q, roi_left, roi_right = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right, gray_left.shape[::-1], R, T)

        # Save calibration results
        calibration_data = {
            'mtx_left': mtx_left, 'dist_left': dist_left,
            'mtx_right': mtx_right, 'dist_right': dist_right,
            'R': R, 'T': T,
            'rect_left': rect_left, 'rect_right': rect_right,
            'proj_mat_left': proj_mat_left, 'proj_mat_right': proj_mat_right,
            'Q': Q
        }

        np.save('stereo_calibration.npy', calibration_data)
        print("Calibration complete. Results saved to 'stereo_calibration.npy'")

    def run_calibration(self):
        try:
            if not self.setup_camera():
                print("Failed to set up the oCams-1CGN-U camera. Exiting.")
                return

            print("Press 'q' to skip image capture and use existing images, if available.")
            print("Press any other key to start capturing new images.")
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or not self.capture_calibration_images():
                print("Checking for existing images...")
                images_left = sorted(glob.glob(os.path.join(self.calibration_folder, 'left_*.jpg')))
                images_right = sorted(glob.glob(os.path.join(self.calibration_folder, 'right_*.jpg')))
                
                if len(images_left) > 0 and len(images_right) > 0:
                    print(f"Found {len(images_left)} image pairs. Proceeding with calibration.")
                    self.calibrate_cameras()
                else:
                    print("No existing images found. Exiting.")
            else:
                self.calibrate_cameras()
        except Exception as e:
            print(f"An error occurred during calibration: {str(e)}")
        finally:
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = StereoCalibrator()
    calibrator.run_calibration()