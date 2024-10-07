# Motion Detection Application

## Overview
This Python application uses OpenCV to detect motion using the computer's camera. It records video clips when motion is detected and stops recording after a defined period of no motion.

## Features
- Continuous video capture from the camera.
- Motion detection with visual indicators.
- Recording of motion events with automatic stopping after a specified duration of no movement.

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
## Usage
1. Run the application:
   ```bash
   python main.py
   ```
2. The application will start capturing video from your camera. It will automatically detect motion and start recording clips when motion is detected.
3. <b>Stop the application:</b> Press `q` at any time to exit the application.

## Configuration
You can adjust the sensitivity parameters in the code to suit your needs:

- **`threshold_value`**: Adjusts the sensitivity for motion detection (default is `20`).
- **`min_contour_area`**: Minimum area (in pixels) to consider as motion (default is `1000`).
- **`no_motion_timeout`**: Time in seconds after which recording stops if no motion is detected (default is `3` seconds).
- **`blur_kernel_size`**: Kernel size for Gaussian blur to reduce noise (default is `(5, 5)`).
- **`frame_skip`**: Number of frames to skip between motion detection (default is `1`).
- **`initial_wait_time`**: Time in seconds to wait before capturing the initial background frame (default is `2` seconds).
