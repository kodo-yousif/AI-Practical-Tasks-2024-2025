# Motion Detection Application

## Overview  
This Python application uses **OpenCV** to detect motion using the computer's camera. It records video clips when motion is detected and stops recording after a defined period of no motion. The application also allows real-time parameter adjustments through trackbars for optimal tuning.

## Features  
- Continuous video capture from the camera.  
- Motion detection with visual indicators (green contours).  
- Automatic recording of motion events with stop functionality after no movement is detected for a set duration.  
- Real-time parameter tuning using **trackbars**.

## Requirements  
- Python 3.x  
- OpenCV (`opencv-python`)  
- NumPy (`numpy`)

## Installation  
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
    ```
2. Install the required packages:
   ```bash
   pip install opencv-python numpy
   ```
## Usage
1. Run the application:
   ```bash
   python main.py
   ```
2. The application will start capturing video from your camera. It will automatically detect motion and start recording clips when motion is detected.
3. <b>Stop the application:</b> Press `q` at any time to exit the application.

## Configuration  
### Adjustable Parameters via Trackbars
- **Threshold Value** (`Threshold`):  
  Controls sensitivity to brightness changes between frames. Default is `20`. Higher values reduce sensitivity.

- **Minimum Contour Area** (`Min Area`):  
  The minimum area (in pixels) for a detected object to be considered as motion. Default is `500`. Increase to ignore smaller movements.

- **Blur Kernel Size** (`Blur Kernel`):  
  Size of the Gaussian blur kernel to reduce noise. Default is `5`. Larger values help suppress small background noise.

- **No Motion Timeout** (`No Motion Timeout`):  
  Time in seconds after which recording stops if no motion is detected. Default is `3` seconds.

- **Background Update Rate** (`BG Update Rate`):  
  Controls how fast the background model is updated, with a value between `0` and `1`. Default is `0.05`.

- **SigmaX for Gaussian Blur** (`SigmaX`):  
  Standard deviation for Gaussian blur on the X-axis. Default is `0`.
