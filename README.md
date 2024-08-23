# Stereo Object Detection System

This project implements a stereo object detection system using YOLOv8 and OpenCV. It includes scripts for camera calibration and real-time object detection with distance estimation.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLOv8

For a complete list of dependencies, see `requirements.txt`.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/stereo-object-detection.git
   cd stereo-object-detection
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Camera Calibration

Before running the object detection, you need to calibrate your stereo camera setup:

1. Run the calibration script:
   ```
   python stereo_camera_calibration.py
   ```

2. Follow the on-screen instructions to capture calibration images.

3. The script will generate a `stereo_calibration.npy` file with the calibration data.

### Object Detection

To run the object detection:
```
python stereo_object_detection.py --stereo
```

Additional options:
- `--simulate`: Simulate stereo camera (for testing without actual stereo hardware)
- `--detect_class CLASS`: Specify the class to detect (default: "person")
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Controls

During object detection:
- Press 'q' to quit
- Press 'c' to change the detection class
- Press 'b' to adjust brightness