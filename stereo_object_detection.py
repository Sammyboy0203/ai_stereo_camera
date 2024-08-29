
# 필요한 라이브러리 임포트
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import io
import sys

class StereoObjectDetector:
    def __init__(self, detect_class='person'):
        # 탐지기 매개변수 초기화
        self.detect_class = detect_class
        self.model = YOLO('yolov8n.pt')  # YOLO 모델 로드
        self.camera = None
        self.distance_scale_factor = 0.1  # 거리 스케일 팩터 추가

        # 카메라 매개변수
        self.frame_width = 2560  # 1280 * 2
        self.frame_height = 960
        self.bytes_per_pixel = 2  # oCamS-1CGN은 16비트 데이터 사용
        self.focal_length = 500  # 픽셀 단위
        self.baseline = 0.1  # 미터 단위 (스테레오 설정용)
        self.max_init_attempts = 5
        self.init_delay = 5

        # 스테레오 카메라 매개변수
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.stereo_map_left = None
        self.stereo_map_right = None
        self.Q = None

        self.load_calibration_data()
        self.create_stereo_matcher()

    def load_calibration_data(self):
        try:
            # 캘리브레이션 데이터 로드
            calibration_data = np.load('stereo_calibration.npy', allow_pickle=True).item()
            self.camera_matrix_left = calibration_data['mtx_left']
            self.camera_matrix_right = calibration_data['mtx_right']
            self.dist_coeffs_left = calibration_data['dist_left']
            self.dist_coeffs_right = calibration_data['dist_right']
            self.R = calibration_data['R']
            self.T = calibration_data['T']
            self.Q = calibration_data['Q']
            
            # 캘리브레이션 데이터에서 초점 거리와 기준선 업데이트
            self.focal_length = self.camera_matrix_left[0, 0]  # fx = fy 가정
            self.baseline = abs(self.T[0])  # 기준선은 T의 X 성분의 절대값
            
            # 스테레오 맵 초기화
            self.stereo_map_left = cv2.initUndistortRectifyMap(self.camera_matrix_left, self.dist_coeffs_left, self.R, self.Q[:3, :3], (self.frame_width, self.frame_height), cv2.CV_16SC2)
            self.stereo_map_right = cv2.initUndistortRectifyMap(self.camera_matrix_right, self.dist_coeffs_right, self.R, self.Q[:3, :3], (self.frame_width, self.frame_height), cv2.CV_16SC2)
            
            print("캘리브레이션 데이터를 성공적으로 로드했습니다")
        except Exception as e:
            print(f"캘리브레이션 데이터 로드 실패: {e}")

    def create_stereo_matcher(self):
        # 스테레오 매칭 객체 생성
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*16,
            blockSize=7,
            P1=8 * 3 * 7**2,
            P2=32 * 3 * 7**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def start_camera(self):
        for attempt in range(self.max_init_attempts):
            print(f"oCamS-1CGN 초기화 시도 중 (시도 {attempt + 1}/{self.max_init_attempts})")
            try:
                # 카메라 열기
                self.camera = cv2.VideoCapture(0)
                
                if not self.camera.isOpened():
                    print("카메라를 열지 못했습니다")
                    time.sleep(self.init_delay)
                    continue

                # 카메라 속성 설정
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, -1)
                self.camera.set(cv2.CAP_PROP_EXPOSURE, -7)
                self.camera.set(cv2.CAP_PROP_GAIN, 100)
                self.camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)

                # 카메라 속성 출력
                print(f"설정된 너비: {self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)}")
                print(f"설정된 높이: {self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                print(f"설정된 밝기: {self.camera.get(cv2.CAP_PROP_BRIGHTNESS)}")
                print(f"설정된 노출: {self.camera.get(cv2.CAP_PROP_EXPOSURE)}")
                print(f"설정된 게인: {self.camera.get(cv2.CAP_PROP_GAIN)}")

                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("oCamS-1CGN에서 프레임을 캡처하지 못했습니다")
                    self.camera.release()
                    time.sleep(self.init_delay)
                    continue

                if frame.shape != (1, 2457600):
                    print(f"예상치 못한 프레임 형태: {frame.shape}")
                    self.camera.release()
                    time.sleep(self.init_delay)
                    continue

                print("oCamS-1CGN을 성공적으로 초기화했습니다")
                return True
            except Exception as e:
                print(f"카메라 설정 중 오류 발생: {str(e)}")
                if self.camera is not None:
                    self.camera.release()
                time.sleep(self.init_delay)

        print("여러 번의 시도 후 카메라 초기화에 실패했습니다")
        return False

    def process_frame(self, frame):
        if frame.shape != (1, 2457600):
            print(f"예상치 못한 프레임 형태: {frame.shape}")
            return None, None

        # 프레임을 2D로 재구성
        frame_2d = frame.reshape((960, 1280 * 2))
        
        # 프레임을 좌우 이미지로 분할
        left = frame_2d[:, 1::2]
        right = frame_2d[:, ::2]
        
        # 베이어에서 BGR로 변환
        try:
            left_bgr = cv2.cvtColor(left, cv2.COLOR_BayerGB2BGR)
            right_bgr = cv2.cvtColor(right, cv2.COLOR_BayerGB2BGR)
        except Exception as e:
            print(f"베이어에서 BGR로 변환 중 오류 발생: {e}")
            return None, None

        return left_bgr, right_bgr

    def estimate_distance(self, bbox_left, bbox_right, left_frame, right_frame):
        try:
            left_rectified = cv2.undistort(left_frame, self.camera_matrix_left, self.dist_coeffs_left)
            right_rectified = cv2.undistort(right_frame, self.camera_matrix_right, self.dist_coeffs_right)
            
            if np.all(left_rectified == 0) or np.all(right_rectified == 0):
                return None
            
            gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
            
            scale_factor = 0.5
            gray_left_small = cv2.resize(gray_left, None, fx=scale_factor, fy=scale_factor)
            gray_right_small = cv2.resize(gray_right, None, fx=scale_factor, fy=scale_factor)
            
            disparity = self.stereo_matcher.compute(gray_left_small, gray_right_small).astype(np.float32) / 16.0
            
            disparity = cv2.resize(disparity, (gray_left.shape[1], gray_left.shape[0]))
            
            x1, y1, x2, y2 = bbox_left
            object_disparity = disparity[y1:y2, x1:x2]
            
            valid_disparities = object_disparity[(object_disparity > 0) & (object_disparity < 128)]
            
            if len(valid_disparities) == 0:
                return None
            
            median_disparity = np.median(valid_disparities)
            
            if median_disparity <= 0:
                return None
            
            distance = (self.focal_length * self.baseline) / median_disparity
            distance = distance * self.distance_scale_factor

            if not np.isfinite(distance):
                return None
            
            return round(float(distance), 2)
        except Exception as e:
            return None

    def detect_objects(self, left_frame, right_frame):
        # Redirect stdout to suppress YOLO output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # YOLOv8을 사용하여 객체 감지 구현 (낮은 신뢰도 임계값 사용)
            results_left = self.model(left_frame, conf=0.3)
            results_right = self.model(right_frame, conf=0.3)
            
            detections = []
            
            for r_left, r_right in zip(results_left, results_right):
                boxes_left = r_left.boxes
                boxes_right = r_right.boxes
                
                # 좌측 프레임 감지 처리
                for box_left in boxes_left:
                    cls_left = self.model.names[int(box_left.cls[0])]
                    if cls_left == self.detect_class:
                        confidence_left = round(float(box_left.conf[0]), 2)
                        bbox_left = list(map(int, box_left.xyxy[0]))
                        
                        detections.append({
                            'class': self.detect_class,
                            'confidence': confidence_left,
                            'bbox_left': bbox_left,
                            'bbox_right': None,
                            'distance': None
                        })
                
                # 우측 프레임 감지 처리
                for box_right in boxes_right:
                    cls_right = self.model.names[int(box_right.cls[0])]
                    if cls_right == self.detect_class:
                        confidence_right = round(float(box_right.conf[0]), 2)
                        bbox_right = list(map(int, box_right.xyxy[0]))
                        
                        # 좌측 감지와 일치시도
                        matched = False
                        for detection in detections:
                            if detection['bbox_right'] is None:
                                detection['bbox_right'] = bbox_right
                                detection['confidence'] = (detection['confidence'] + confidence_right) / 2
                                matched = True
                                break
                        
                        # 일치하는 감지가 없으면 새로운 감지로 추가
                        if not matched:
                            detections.append({
                                'class': self.detect_class,
                                'confidence': confidence_right,
                                'bbox_left': None,
                                'bbox_right': bbox_right,
                                'distance': None
                            })
            
            # 일치한 감지에 대해 거리 추정
            for detection in detections:
                if detection['bbox_left'] and detection['bbox_right']:
                    detection['distance'] = self.estimate_distance(detection['bbox_left'], detection['bbox_right'], left_frame, right_frame)
            
            return detections
        finally:
            # Restore stdout
            sys.stdout = old_stdout

    def adjust_camera_settings(self, brightness=None, exposure=None, gain=None):
        if self.camera is not None:
            if brightness is not None:
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            if exposure is not None:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            if gain is not None:
                self.camera.set(cv2.CAP_PROP_GAIN, gain)
            
            print(f"카메라 설정 업데이트:")
            print(f"밝기: {self.camera.get(cv2.CAP_PROP_BRIGHTNESS)}")
            print(f"노출: {self.camera.get(cv2.CAP_PROP_EXPOSURE)}")
            print(f"게인: {self.camera.get(cv2.CAP_PROP_GAIN)}")

    def run(self):
        if not self.start_camera():
            print("카메라 시작 실패. 종료.")
            return

        print(f"탐지 시작. 'q'를 눌러 종료, 'c'를 눌러 클래스 변경, 'b'를 눌러 밝기 조절, 'e'를 눌러 노출 조절, 'g'를 눌러 게인 조절.")
        
        while True:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    print("프레임 캡처 실패")
                    break

                left_bgr, right_bgr = self.process_frame(frame)
                if left_bgr is None or right_bgr is None:
                    continue
                
                detections = self.detect_objects(left_bgr, right_bgr)
                
                for detection in detections:
                    cls = detection['class']
                    confidence = detection['confidence']
                    bbox_left = detection['bbox_left']
                    bbox_right = detection['bbox_right']
                    distance = detection['distance']
                    
                    # Draw bounding boxes and information on frames
                    if bbox_left:
                        x1, y1, x2, y2 = bbox_left
                        cv2.rectangle(left_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(left_bgr, f"{cls} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        if distance is not None:
                            cv2.putText(left_bgr, f"Dist: {distance}m", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.putText(left_bgr, "Dist: N/A", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    if bbox_right:
                        x1, y1, x2, y2 = bbox_right
                        cv2.rectangle(right_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(right_bgr, f"{cls} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        if distance is not None:
                            cv2.putText(right_bgr, f"Dist: {distance}m", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.putText(right_bgr, "Dist: N/A", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                cv2.imshow("Left Camera", left_bgr)
                cv2.imshow("Right Camera", right_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.detect_class = input("탐지할 클래스 입력: ")
                    print(f"이제 탐지: {self.detect_class}")
                elif key == ord('b'):
                    brightness = float(input("새로운 밝기 값 입력 (-64에서 64까지): "))
                    self.adjust_camera_settings(brightness=brightness)
                elif key == ord('e'):
                    exposure = float(input("새로운 노출 값 입력 (-13에서 -1까지): "))
                    self.adjust_camera_settings(exposure=exposure)
                elif key == ord('g'):
                    gain = float(input("새로운 게인 값 입력 (0에서 240까지): "))
                    self.adjust_camera_settings(gain=gain)

            except Exception as e:
                print(f"오류 발생: {str(e)}")
                break

        self.camera.release()
        cv2.destroyAllWindows()

def main():
    # 명령줄 인수 처리 및 초기화
    parser = argparse.ArgumentParser(description="oCamS-1CGN 스테레오 객체 탐지 데모")
    parser.add_argument("--detect_class", type=str, default="person", help="탐지할 클래스")
    args = parser.parse_args()

    detector = StereoObjectDetector(detect_class=args.detect_class)
    detector.run()

if __name__ == "__main__":
    main()