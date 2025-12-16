import operations

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
    result = image.copy()
    for step in steps:
        func = OPERATION_MAP.get(step)
        if func:
            result = func(result)
    return result
