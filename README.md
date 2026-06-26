# Finger Filters

Real-time webcam filter controlled by hand gestures. Uses **MediaPipe** hand tracking and **OpenCV** filters.

## How it works

1. Show both hands to the camera
2. Raise fingers on your **left hand** to switch filters
3. Filters are applied to the regions between your fingertips

| Fingers | Filter |
|---------|--------|
| 0 (fist) | No filter |
| 1 | Grayscale |
| 2 | Canny Edges |
| 3 | Gaussian Blur |
| 4 | Cartoon |
| 5 | Invert |

## Quick Start (Windows EXE)

Download `FingerFilters.exe` from the [Releases](https://github.com/Polarzen/finger-filters/releases) page (or `dist/FingerFilters.exe`). Double-click to run. No Python or setup required — just a webcam.

Press **Q** to quit.

## Setup (Python)

```bash
# Clone the repo
git clone git@github.com:Polarzen/finger-filters.git
cd finger-filters

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python finger_filters.py
```

Press **Q** to quit.