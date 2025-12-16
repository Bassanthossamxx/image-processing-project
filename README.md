# Image Processing Application

A beginner-friendly GUI application for applying various image processing filters and effects using Python, OpenCV, and Tkinter.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Usage Guide](#-usage-guide)
- [Available Filters](#-available-filters)
- [Documentation](#-documentation)

---

## Features

| Feature | Description |
|---------|-------------|
| **10 Image Filters** | Histogram equalization, blur, sharpen, noise, and more |
| **Pipeline System** | Combine 2-4 filters in sequence |
| **Auto-Save** | Automatically saves processed images with timestamps |
| **Simple GUI** | Easy-to-use Tkinter interface |
| **Organized Output** | Each filter saves to its own folder |

---

## Project Structure

```
image-processing-project/
│
├── image_processing_app/       # Main application folder
│   ├── main.py                 # Entry point - GUI application
│   ├── operations.py           # All image processing functions
│   ├── pipeline.py             # Pipeline management system
│   ├── saver.py                # Auto-save functionality
│   └── output/                 # Saved processed images
│
├── requirements.txt            # Python dependencies
├── PROJECT_DOCUMENTATION.md    # Detailed documentation
├── LICENSE                     # MIT License
└── README.md                   # This file
```

### File Overview

| File | Purpose |
|------|---------|
| **main.py** | The main GUI application. Creates the window, buttons, and handles user interactions. Run this file to start the app. |
| **operations.py** | Contains all 10 image processing functions (filters). Each function takes an image and returns a processed image. |
| **pipeline.py** | Manages the pipeline feature. Maps operation names to functions and applies multiple filters in sequence. |
| **saver.py** | Handles automatic saving. Creates organized folders and saves images with timestamps. |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd image-processing-project

# Or simply download and extract the ZIP file
```

### Step 2: Install Dependencies

Open a terminal/command prompt in the project folder and run:

```bash
pip install -r requirements.txt
```

This installs:
- **opencv-python** - Image processing library
- **numpy** - Numerical operations
- **Pillow** - Image display in GUI

---

## How to Run

### Option 1: From Terminal (Recommended)

```bash
# Navigate to the app folder
cd image_processing_app

# Run the application
python main.py
```

### Option 2: From VS Code

1. Open the project folder in VS Code
2. Open `image_processing_app/main.py`
3. Press `F5` or click "Run Python File"

### Option 3: Double-Click (Windows)

1. Navigate to `image_processing_app` folder
2. Double-click `main.py` (if Python is set as default for .py files)

---

## Usage Guide

### Basic Workflow

1. **Upload an Image** → Click "Upload Image" and select a photo
2. **Choose a Filter** → Select from the dropdown menu
3. **Apply** → Click "Apply" to process the image
4. **View Result** → See the processed image on the right
5. **Save** → Image is auto-saved, or click "Save to Desktop"

### Using the Pipeline

1. Click **"Choose Steps"** to open the selection window
2. **Check 2-4 filters** you want to apply
3. Click **"Apply"** in the selection window
4. Click **"Run Pipeline"** to process the image

---

## Available Filters

### Image Enhancement
| Filter | Effect |
|--------|--------|
| `histogram` | Improves contrast by spreading pixel intensities |
| `smooth` | Blurs the image using Gaussian filter |
| `sharpen` | Enhances edges and details |

### Noise Operations
| Filter | Effect |
|--------|--------|
| `gaussian_noise` | Adds random Gaussian noise |
| `salt_pepper` | Adds black and white dots |
| `median` | Removes salt & pepper noise |

### Color Processing
| Filter | Effect |
|--------|--------|
| `grayscale` | Converts to black and white |
| `color_enhance` | Increases brightness |

### Frequency Domain
| Filter | Effect |
|--------|--------|
| `lowpass` | Keeps smooth areas, removes details |
| `highpass` | Keeps edges, removes smooth areas |

---

## Documentation

For detailed explanations of each filter's theory and code, see:

**[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**

This includes:
- Mathematical formulas
- Step-by-step code explanations
- Visual diagrams
- Presentation-ready content