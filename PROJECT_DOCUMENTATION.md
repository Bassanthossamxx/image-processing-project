# 📚 Image Processing Application - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Image Enhancement Filters](#image-enhancement-filters)
   - [Histogram Equalization](#1-histogram-equalization)
   - [Gaussian Smoothing](#2-gaussian-smoothing-blur)
   - [Laplacian Sharpening](#3-laplacian-sharpening)
3. [Noise & Restoration](#noise--restoration)
   - [Gaussian Noise](#4-gaussian-noise)
   - [Salt & Pepper Noise](#5-salt--pepper-noise)
   - [Median Filter](#6-median-filter)
4. [Color Processing](#color-processing)
   - [Grayscale Conversion](#7-grayscale-conversion)
   - [Color Enhancement](#8-color-enhancement-brightness)
5. [Frequency Domain Filters](#frequency-domain-filters)
   - [Low-Pass Filter](#9-low-pass-filter)
   - [High-Pass Filter](#10-high-pass-filter)
6. [Pipeline System](#pipeline-system)
7. [Auto-Save & Manual Save](#auto-save--manual-save-system)
8. [Main Application (GUI)](#main-application-gui)
9. [Quick Reference Summary](#quick-reference-summary)

---

## Project Overview

This Image Processing Application is a GUI-based tool built with Python that allows users to apply various image processing operations. The application uses:

| Library | Purpose |
|---------|---------|
| **OpenCV (cv2)** | Core image processing operations |
| **NumPy** | Numerical operations on arrays |
| **Tkinter** | Graphical User Interface |
| **PIL (Pillow)** | Image display in GUI |

### Project Structure
```
image_processing_app/
├── main.py          → GUI application (user interface)
├── operations.py    → All image processing functions (filters)
├── pipeline.py      → Pipeline management (combine multiple filters)
├── saver.py         → Auto-save functionality
└── output/          → Saved processed images
```

---

# Image Enhancement Filters

## 1. Histogram Equalization

### 📖 Theory

**What is a Histogram?**
A histogram is a graph showing the distribution of pixel intensity values in an image. For a grayscale image:
- X-axis: Pixel intensity (0 = black, 255 = white)
- Y-axis: Number of pixels with that intensity

**What is Histogram Equalization?**
Histogram equalization is a technique to improve the **contrast** of an image by spreading out the most frequent intensity values across the full range (0-255).

**Why use it?**
- Images with low contrast have a narrow histogram (pixels concentrated in a small range)
- Histogram equalization spreads pixels across the full range
- Result: Better visibility of details, enhanced contrast

**Mathematical Concept:**
The transformation function uses the Cumulative Distribution Function (CDF):

$$s = T(r) = (L-1) \cdot CDF(r)$$

Where:
- $r$ = input pixel value
- $s$ = output pixel value
- $L$ = number of intensity levels (256)
- $CDF$ = cumulative distribution function of pixel intensities

### 💻 Code Explanation

```python
def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Step 1: Convert to grayscale
    eq = cv2.equalizeHist(gray)                     # Step 2: Apply histogram equalization
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)     # Step 3: Convert back to BGR
```

**Step-by-step:**
1. **`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`** - Converts color image to grayscale because histogram equalization works on single-channel images
2. **`cv2.equalizeHist(gray)`** - OpenCV's built-in function that:
   - Calculates the histogram of the image
   - Computes the cumulative distribution function (CDF)
   - Remaps pixel values to spread them evenly
3. **`cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)`** - Converts back to 3-channel image for display consistency

---

## 2. Gaussian Smoothing (Blur)

### 📖 Theory

**What is Gaussian Blur?**
Gaussian blur is a smoothing filter that reduces image noise and detail by averaging each pixel with its neighbors using a Gaussian (bell-shaped) weighted kernel.

**Gaussian Function (2D):**

$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

Where:
- $(x, y)$ = distance from the center pixel
- $\sigma$ = standard deviation (controls blur amount)

**How the Kernel Works:**
For a 5×5 kernel, weights look like:
```
[1  4  7  4  1]
[4 16 26 16  4]
[7 26 41 26  7]  ÷ 273 (normalized)
[4 16 26 16  4]
[1  4  7  4  1]
```

The center pixel gets the highest weight, and weights decrease as distance increases (Gaussian distribution).

**Why use it?**
- Reduces noise (removes random pixel variations)
- Smooths edges (preprocessing for edge detection)
- Creates blur effect

### 💻 Code Explanation

```python
def smooth_gaussian(img, ksize=5):
    if ksize % 2 == 0:      # Step 1: Ensure kernel size is odd
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)  # Step 2: Apply Gaussian blur
```

**Step-by-step:**
1. **`if ksize % 2 == 0: ksize += 1`** - Kernel size must be ODD (3, 5, 7, etc.) because:
   - The kernel needs a center pixel
   - OpenCV requires odd kernel sizes
2. **`cv2.GaussianBlur(img, (ksize, ksize), 0)`**:
   - `img` = input image
   - `(ksize, ksize)` = kernel dimensions (5×5 by default)
   - `0` = sigma (standard deviation), when 0, OpenCV calculates it automatically from kernel size

---

## 3. Laplacian Sharpening

### 📖 Theory

**What is Sharpening?**
Sharpening enhances edges and fine details in an image by emphasizing high-frequency components (rapid intensity changes).

**The Laplacian Operator:**
The Laplacian is a second-order derivative operator that detects edges:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

**Laplacian Kernel:**
```
[0  1  0]
[1 -4  1]
[0  1  0]
```

**Sharpening Formula:**

$$\text{Sharpened} = \text{Original} - \text{Laplacian}$$

By subtracting the Laplacian (which captures edges), we enhance the edges.

**Why use it?**
- Makes edges more pronounced
- Increases perceived clarity
- Useful for blurry images

### 💻 Code Explanation

```python
def sharpen_laplacian(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Step 1: Convert to grayscale
    lap = cv2.Laplacian(gray, cv2.CV_64F)           # Step 2: Compute Laplacian
    sharp = gray - lap                               # Step 3: Subtract Laplacian
    sharp = np.clip(sharp, 0, 255).astype(np.uint8) # Step 4: Clip values
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)  # Step 5: Convert back to BGR
```

**Step-by-step:**
1. **Convert to grayscale** - Laplacian works on single-channel images
2. **`cv2.Laplacian(gray, cv2.CV_64F)`**:
   - Computes second derivative (edge detection)
   - `cv2.CV_64F` = 64-bit float to handle negative values
3. **`gray - lap`** - Subtracting Laplacian enhances edges
4. **`np.clip(sharp, 0, 255)`** - Ensures values stay in valid range [0, 255]
5. **Convert back to BGR** - For display consistency

---

# Noise & Restoration

## 4. Gaussian Noise

### 📖 Theory

**What is Gaussian Noise?**
Gaussian noise is random variation in pixel intensity values that follows a normal (Gaussian) distribution. It's the most common type of noise in digital images.

**Gaussian Distribution:**

$$P(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

Where:
- $\mu$ = mean (usually 0 for noise)
- $\sigma$ = standard deviation (controls noise intensity)

**Why simulate noise?**
- To test denoising algorithms
- To understand noise effects
- Common in low-light photography, sensor noise

### 💻 Code Explanation

```python
def add_gaussian_noise(img, level=20):
    noise = np.random.normal(0, level, img.shape).astype(np.int16)  # Step 1: Generate noise
    noisy = img.astype(np.int16) + noise                             # Step 2: Add noise
    return np.clip(noisy, 0, 255).astype(np.uint8)                   # Step 3: Clip values
```

**Step-by-step:**
1. **`np.random.normal(0, level, img.shape)`**:
   - Generates random values from Gaussian distribution
   - Mean = 0 (noise equally positive/negative)
   - Standard deviation = level (20 by default, controls noise amount)
   - Shape matches image dimensions
   - `.astype(np.int16)` - Signed integer to handle negative values
2. **`img.astype(np.int16) + noise`** - Adds noise to each pixel
3. **`np.clip(noisy, 0, 255)`** - Ensures valid range, then converts to uint8

---

## 5. Salt & Pepper Noise

### 📖 Theory

**What is Salt & Pepper Noise?**
Salt and pepper noise (also called impulse noise) randomly sets pixels to either maximum (255 = white = "salt") or minimum (0 = black = "pepper") values.

**Characteristics:**
- Random white dots (salt) and black dots (pepper)
- Often caused by transmission errors, sensor malfunction
- Probability $p$ controls density of noise

**Mathematical Model:**
For each pixel with probability $p$:

$$\text{pixel} = \begin{cases} 0 & \text{with probability } p/2 \\ 255 & \text{with probability } p/2 \\ \text{original} & \text{with probability } 1-p \end{cases}$$

### 💻 Code Explanation

```python
def add_salt_pepper_noise(img, prob=0.02):
    noisy = img.copy()                              # Step 1: Copy image
    rnd = np.random.rand(*img.shape[:2])            # Step 2: Generate random values
    noisy[rnd < prob] = 0                           # Step 3: Add pepper (black)
    noisy[rnd > 1 - prob] = 255                     # Step 4: Add salt (white)
    return noisy
```

**Step-by-step:**
1. **`img.copy()`** - Creates a copy to avoid modifying original
2. **`np.random.rand(*img.shape[:2])`**:
   - Generates random values between 0 and 1
   - Shape is image height × width (2D)
   - `*` unpacks the tuple
3. **`noisy[rnd < prob] = 0`** - Where random < 0.02, set pixel to 0 (black/pepper)
4. **`noisy[rnd > 1 - prob] = 255`** - Where random > 0.98, set pixel to 255 (white/salt)

---

## 6. Median Filter

### 📖 Theory

**What is Median Filtering?**
Median filter is a non-linear filter that replaces each pixel with the **median** value of neighboring pixels. It's excellent for removing salt & pepper noise while preserving edges.

**How it works:**
1. Take a neighborhood (e.g., 5×5) around each pixel
2. Sort all pixel values in the neighborhood
3. Replace center pixel with the median (middle value)

**Example (3×3):**
```
Neighborhood:     Sorted: [120, 130, 140, 145, 150, 155, 160, 170, 255]
[120, 140, 160]                              ↑
[130, 255, 170]   →   Median = 150 (middle value)
[150, 145, 155]
```
The outlier (255, salt noise) is eliminated!

**Why Median Filter?**
- Best for salt & pepper noise
- Preserves edges (unlike Gaussian blur)
- Non-linear (doesn't just average)

### 💻 Code Explanation

```python
def median_filter(img, ksize=5):
    if ksize % 2 == 0:      # Step 1: Ensure odd kernel size
        ksize += 1
    return cv2.medianBlur(img, ksize)  # Step 2: Apply median filter
```

**Step-by-step:**
1. **Odd kernel size** - Required for having a center pixel
2. **`cv2.medianBlur(img, ksize)`**:
   - Takes each pixel's 5×5 neighborhood
   - Finds the median of all 25 values
   - Replaces center pixel with median

---

# Color Processing

## 7. Grayscale Conversion

### 📖 Theory

**What is Grayscale?**
Grayscale is a single-channel image where each pixel represents intensity from black (0) to white (255), with no color information.

**Conversion Formula:**
The standard luminosity method weights RGB channels by human eye sensitivity:

$$Gray = 0.299 \times R + 0.587 \times G + 0.114 \times B$$

- **Green** has highest weight because human eyes are most sensitive to green
- **Blue** has lowest weight because eyes are least sensitive to blue

**Why convert to grayscale?**
- Reduces data (1 channel vs 3 channels)
- Required for many algorithms (edge detection, etc.)
- Simplifies processing

### 💻 Code Explanation

```python
def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # Step 1: Convert to grayscale
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Step 2: Convert back to 3-channel
```

**Step-by-step:**
1. **`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`** - Applies luminosity formula to convert to single channel
2. **`cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)`** - Converts back to 3 channels (all channels have same value) for display consistency with other operations

---

## 8. Color Enhancement (Brightness)

### 📖 Theory

**HSV Color Space:**
HSV (Hue, Saturation, Value) separates color information from brightness:
- **Hue (H)**: Color type (0-180 in OpenCV)
- **Saturation (S)**: Color intensity/purity (0-255)
- **Value (V)**: Brightness (0-255)

**Why use HSV for brightness?**
In RGB, changing brightness means changing all three channels proportionally. In HSV, we can change just the V (Value) channel without affecting the color.

**Enhancement Process:**
1. Convert BGR → HSV
2. Add value to V channel
3. Convert HSV → BGR

### 💻 Code Explanation

```python
def color_enhancement(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)     # Step 1: Convert to HSV
    h, s, v = cv2.split(hsv)                        # Step 2: Split channels
    v = cv2.add(v, value)                           # Step 3: Increase brightness
    hsv = cv2.merge((h, s, v))                      # Step 4: Merge channels
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)     # Step 5: Convert back to BGR
```

**Step-by-step:**
1. **`cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`** - Converts color space
2. **`cv2.split(hsv)`** - Separates into 3 individual channels
3. **`cv2.add(v, value)`** - Adds 30 to all V values (automatic clipping at 255)
4. **`cv2.merge((h, s, v))`** - Combines channels back
5. **Convert back to BGR** - For display

---

# Frequency Domain Filters

## Understanding Frequency Domain

### 📖 Theory

**What is Frequency in Images?**
In images, frequency refers to how quickly pixel values change:
- **Low frequency**: Smooth areas, gradual changes (sky, walls)
- **High frequency**: Sharp changes, edges, details, noise

**Fourier Transform:**
The 2D Fourier Transform converts an image from spatial domain (pixels) to frequency domain (frequencies):

$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}$$

**Frequency Domain Image:**
- Center = Low frequencies (smooth areas)
- Edges = High frequencies (details, edges)

---

## 9. Low-Pass Filter

### 📖 Theory

**What is a Low-Pass Filter?**
A low-pass filter keeps low frequencies and removes high frequencies. This results in:
- Blurring effect
- Noise reduction
- Smoothing

**How it works:**
1. Convert image to frequency domain (FFT)
2. Create a mask that keeps center (low frequencies)
3. Apply mask (multiply)
4. Convert back to spatial domain (inverse FFT)

**Ideal Low-Pass Filter:**

$$H(u,v) = \begin{cases} 1 & \text{if } D(u,v) \leq \text{cutoff} \\ 0 & \text{otherwise} \end{cases}$$

Where $D(u,v)$ is distance from center.

### 💻 Code Explanation

```python
def low_pass_filter(img, cutoff=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # Step 1: Convert to grayscale
    f = np.fft.fftshift(np.fft.fft2(gray))                # Step 2: FFT and shift
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2                     # Step 3: Find center

    mask = np.zeros((rows, cols), np.uint8)               # Step 4: Create mask
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1  # Step 5: Square pass region

    f_filtered = f * mask                                  # Step 6: Apply mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))  # Step 7: Inverse FFT
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)
```

**Step-by-step:**
1. **Convert to grayscale** - FFT typically applied to single channel
2. **`np.fft.fft2(gray)`** - 2D Fast Fourier Transform
   **`np.fft.fftshift(...)`** - Shifts zero frequency (DC) to center
3. **Find center coordinates** - Where low frequencies are after shift
4. **Create zero mask** - Start with all zeros (block everything)
5. **Set center region to 1** - Creates a square of size (2×cutoff) × (2×cutoff) that passes through
6. **Multiply** - Element-wise multiplication blocks high frequencies
7. **Inverse FFT**:
   - `np.fft.ifftshift()` - Shift back
   - `np.fft.ifft2()` - Inverse FFT
   - `np.abs()` - Take magnitude (discard complex part)

---

## 10. High-Pass Filter

### 📖 Theory

**What is a High-Pass Filter?**
A high-pass filter keeps high frequencies and removes low frequencies. This results in:
- Edge enhancement
- Detail extraction
- Removes smooth areas (appears darker)

**How it works:**
Opposite of low-pass - block the center, keep the edges of frequency domain.

**Ideal High-Pass Filter:**

$$H(u,v) = \begin{cases} 0 & \text{if } D(u,v) \leq \text{cutoff} \\ 1 & \text{otherwise} \end{cases}$$

### 💻 Code Explanation

```python
def high_pass_filter(img, cutoff=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # Step 1: Convert to grayscale
    f = np.fft.fftshift(np.fft.fft2(gray))                # Step 2: FFT and shift
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2                     # Step 3: Find center

    mask = np.ones((rows, cols), np.uint8)                # Step 4: Create mask (all ones)
    mask[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 0  # Step 5: Block center

    f_filtered = f * mask                                  # Step 6: Apply mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))  # Step 7: Inverse FFT
    return cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)
```

**Key Difference from Low-Pass:**
- **Low-Pass**: `mask = zeros`, then set center to 1 (keep center)
- **High-Pass**: `mask = ones`, then set center to 0 (block center)

---

# Pipeline System

## 📖 Theory

**What is a Pipeline?**
A pipeline is a sequence of operations applied one after another. The output of each operation becomes the input of the next.

```
Image → [Filter 1] → [Filter 2] → [Filter 3] → Final Result
```

**Benefits:**
- Combine multiple effects
- Reproducible processing
- Flexible workflow

### 💻 Code Explanation

**pipeline.py:**

```python
import operations

# Dictionary mapping operation names to functions
OPERATION_MAP = {
    "histogram": operations.histogram_equalization,
    "smooth": operations.smooth_gaussian,
    "sharpen": operations.sharpen_laplacian,
    "gaussian_noise": operations.add_gaussian_noise,
    "salt_pepper": operations.add_salt_pepper_noise,
    "median": operations.median_filter,
    "grayscale": operations.to_grayscale,
    "color_enhance": operations.color_enhancement,
    "lowpass": operations.low_pass_filter,
    "highpass": operations.high_pass_filter,
}

def apply_pipeline(image, steps):
    result = image.copy()           # Don't modify original
    for step in steps:              # Loop through each step
        func = OPERATION_MAP.get(step)  # Get the function
        if func:
            result = func(result)   # Apply and update result
    return result
```

**How it works:**
1. **`OPERATION_MAP`** - Dictionary that maps string names to actual functions
2. **`apply_pipeline(image, steps)`**:
   - Takes an image and list of step names
   - Loops through each step
   - Looks up the function in the dictionary
   - Applies function to current result
   - Returns final processed image

**Example Pipeline Flow:**
```python
steps = ["grayscale", "smooth", "lowpass"]

# Step 1: grayscale
result = to_grayscale(original_image)

# Step 2: smooth  
result = smooth_gaussian(result)  # Uses output from step 1

# Step 3: lowpass
result = low_pass_filter(result)  # Uses output from step 2

# Final result has all three effects applied
```

**GUI Pipeline Selection:**
- User clicks "Choose Steps" to open selection window
- Checkboxes allow selecting 2-4 operations
- "Run Pipeline" applies all selected operations in order

---

# Auto-Save & Manual Save System

## 📖 Theory

**Auto-Save:**
Every time an operation is applied, the result is automatically saved to a designated folder with a timestamp. This ensures:
- No work is lost
- History of all operations
- Easy comparison of results

**Manual Save:**
User can also manually save to a location of their choice (like Desktop).

### 💻 Code Explanation

**saver.py:**

```python
import os
import cv2
from datetime import datetime

BASE_DIR = "output"  # Base folder for all saved images

def save_image(image, operation_name):
    os.makedirs(BASE_DIR, exist_ok=True)                    # Step 1: Create base folder

    folder = os.path.join(BASE_DIR, operation_name)         # Step 2: Create operation folder
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # Step 3: Generate timestamp
    filename = f"{operation_name}_{timestamp}.png"          # Step 4: Create filename
    path = os.path.join(folder, filename)                   # Step 5: Full path

    cv2.imwrite(path, image)                                # Step 6: Save image
```

**Step-by-step:**
1. **`os.makedirs(BASE_DIR, exist_ok=True)`** - Creates "output" folder if it doesn't exist
2. **Creates subfolder** - Each operation gets its own folder (e.g., `output/histogram/`)
3. **`datetime.now().strftime("%Y%m%d_%H%M%S")`** - Creates timestamp like "20241216_143052"
4. **Filename format** - `operation_timestamp.png` (e.g., `histogram_20241216_143052.png`)
5. **Full path** - Combines folder and filename
6. **`cv2.imwrite()`** - Saves the image

**Folder Structure Created:**
```
output/
├── histogram/
│   ├── histogram_20241216_143052.png
│   └── histogram_20241216_143115.png
├── smooth/
│   └── smooth_20241216_143200.png
├── pipeline/
│   └── pipeline_20241216_143230.png
└── ...
```

**When Auto-Save is Triggered (in main.py):**

```python
def apply_operation(self):
    # ... apply the operation ...
    saver.save_image(self.processed_img, op)  # Auto-save after applying

def run_pipeline(self):
    # ... run pipeline ...
    saver.save_image(self.processed_img, "pipeline")  # Auto-save pipeline result
```

**Manual Save to Desktop:**

```python
def save_to_desktop(self):
    if self.processed_img is None:
        messagebox.showwarning("No Image", "No processed image to save.")
        return

    desktop = os.path.join(os.path.expanduser("~"), "Desktop")  # Get desktop path
    filename = filedialog.asksaveasfilename(
        initialdir=desktop,                    # Start at desktop
        defaultextension=".png",               # Default to PNG
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")],
        initialfile="processed_image.png"      # Suggested filename
    )
    if filename:
        cv2.imwrite(filename, self.processed_img)
        messagebox.showinfo("Saved", f"Image saved to:\n{filename}")
```

---

# Main Application (GUI)

### 💻 Complete Flow Explanation

**main.py Structure:**

```python
class App:
    def __init__(self, root):
        # Initialize window, buttons, labels
        # Set up image display areas
        # Create operation dropdown and buttons
```

**Key Components:**

| Component | Purpose |
|-----------|---------|
| `self.original_img` | Stores the uploaded image |
| `self.processed_img` | Stores the result after processing |
| `self.label1` | Displays original image |
| `self.label2` | Displays processed image |
| `self.operation` | Current selected operation |
| `self.selected_steps` | List of pipeline steps |

**Application Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Image Processing App                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌──────────────┐          ┌──────────────┐               │
│    │   Original   │          │   Processed  │               │
│    │    Image     │          │    Image     │               │
│    └──────────────┘          └──────────────┘               │
│                                                              │
│  [Upload Image] [▼ histogram] [Apply] [Save to Desktop]     │
│                                                              │
│         Pipeline: grayscale, smooth, lowpass                 │
│         [Choose Steps] [Run Pipeline]                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**User Actions:**

1. **Upload Image** → Calls `load_image()` → Opens file dialog → Loads image → Displays in left panel

2. **Select Operation** → Dropdown menu → Sets `self.operation` variable

3. **Apply** → Calls `apply_operation()`:
   - Gets selected operation name
   - Looks up function in OPERATION_MAP
   - Applies function to original image
   - Displays result in right panel
   - **Auto-saves** to output folder

4. **Choose Steps** → Opens new window with checkboxes → User selects 2-4 operations → Confirms selection

5. **Run Pipeline** → Calls `run_pipeline()`:
   - Validates 2-4 steps selected
   - Calls `pipeline.apply_pipeline()`
   - Displays result
   - **Auto-saves** to output/pipeline/

6. **Save to Desktop** → Opens save dialog → User chooses location → Saves processed image

---

# Quick Reference Summary

## Filter Comparison Table

| Filter | Category | Effect | Best For |
|--------|----------|--------|----------|
| Histogram Equalization | Enhancement | Improves contrast | Low contrast images |
| Gaussian Smooth | Enhancement | Blurs/smooths | Noise reduction |
| Laplacian Sharpen | Enhancement | Enhances edges | Blurry images |
| Gaussian Noise | Noise | Adds random noise | Testing denoising |
| Salt & Pepper | Noise | Adds black/white dots | Testing denoising |
| Median Filter | Restoration | Removes impulse noise | Salt & pepper removal |
| Grayscale | Color | Removes color | Preprocessing |
| Color Enhance | Color | Increases brightness | Dark images |
| Low-Pass | Frequency | Blurs (keeps smooth) | Smoothing |
| High-Pass | Frequency | Edge detection | Finding edges |

## Key Formulas

| Operation | Formula/Concept |
|-----------|-----------------|
| Grayscale | $0.299R + 0.587G + 0.114B$ |
| Gaussian | $G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$ |
| Sharpening | $\text{Sharp} = \text{Original} - \text{Laplacian}$ |
| Histogram Eq. | $s = (L-1) \cdot CDF(r)$ |

## Libraries Used

| Library | Import | Purpose |
|---------|--------|---------|
| OpenCV | `import cv2` | Image processing |
| NumPy | `import numpy as np` | Array operations |
| Tkinter | `import tkinter as tk` | GUI |
| PIL | `from PIL import Image, ImageTk` | Display images in Tkinter |

---

## 🎯 Key Points for Presentation

1. **Spatial vs Frequency Domain:**
   - Spatial: Operate directly on pixels (Gaussian blur, median filter)
   - Frequency: Transform to frequencies, filter, transform back (low-pass, high-pass)

2. **Linear vs Non-Linear Filters:**
   - Linear: Gaussian blur (weighted average)
   - Non-Linear: Median filter (uses sorting, not averaging)

3. **Edge Preservation:**
   - Gaussian blur smooths edges
   - Median filter preserves edges while removing noise

4. **Pipeline Advantage:**
   - Combine multiple effects
   - Order matters! (e.g., denoise before sharpening)

5. **Auto-Save Benefits:**
   - Never lose work
   - Track processing history
   - Compare different parameters
