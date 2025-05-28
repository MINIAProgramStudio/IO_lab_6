import os
import numpy as np
from PIL import Image
from tqdm.contrib.concurrent import process_map


def rgb_to_hsv_to_label_map(img):
    """
    Convert RGB image to HSV and generate a label map based on conditions.

    Args:
        img (np.ndarray): RGB image array with values in [0,1], shape (height, width, 3).

    Returns:
        np.ndarray: Label map with integer labels (0-8) for each pixel, shape (height, width).
    """
    # RGB to HSV conversion
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    cmax = np.max(img, axis=-1)
    cmin = np.min(img, axis=-1)
    delta = cmax - cmin

    # Hue calculation
    hue = np.zeros_like(cmax)
    mask = delta > 0
    r_mask = (cmax == r) & mask
    g_mask = (cmax == g) & mask
    b_mask = (cmax == b) & mask

    hue[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
    hue[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
    hue[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4
    hue /= 6  # Normalize to [0,1]

    # Saturation and value
    saturation = np.zeros_like(cmax)
    saturation[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
    value = cmax

    # Label map initialization (default label = 8)
    label_map = np.full(hue.shape, 8, dtype=np.int32)

    # Conditions
    conditions = [
        (saturation < 0.1) & (value > 0.9),                          # light (0)
        value < 0.3,                                                # dark (1)
        (hue < 1 / 12) | (hue > 11 / 12),                           # red (2)
        np.abs(hue - 1 / 3) < 1 / 12,                               # green (3)
        np.abs(hue - 2 / 3) < 1 / 12,                               # blue (4)
        np.abs(hue - 1 / 2) < 1 / 12,                               # cyan (5)
        np.abs(hue - 1 / 8) < 1 / 12,                               # yellow (6)
        np.abs(hue - 5 / 6) < 1 / 12                                # magenta (7)
    ]
    labels = [0, 1, 2, 3, 4, 5, 6, 7]

    # Apply labels with reversed precedence
    for cond, label in zip(conditions[::-1], labels[::-1]):
        label_map[cond] = label

    return label_map


def check_and_delete(file_path, check_label_maps=True):
    try:
        with Image.open(file_path) as img:
            img = img.convert('RGB')  # Ensure image is RGB
            img_array = np.array(img)

            if img_array.ndim != 3 or img_array.shape[2] != 3:
                os.remove(file_path)
                return 1

            if not check_label_maps:
                return 0

            # Normalize to [0,1]
            img_normalized = img_array.astype(np.float32) / 255.0

            # Generate label map
            label_map = rgb_to_hsv_to_label_map(img_normalized)

            # Flatten the label map and count occurrences
            label_map_flat = label_map.flatten()
            counts = np.bincount(label_map_flat, minlength=9)  # Labels 0–8
            max_count = np.max(counts)
            total_pixels = label_map_flat.size

            # Check if the most frequent label exceeds 85%
            if max_count > 0.85 * total_pixels:
                os.remove(file_path)
                return 1

            return 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0


if __name__ == '__main__':
    # Define the directory path containing the images
    dir_path = 'datasets/train2017'  # Replace with your directory path

    # Define common image file extensions
    image_ext = ['.jpg', '.jpeg', '.png', '.gif']

    # Get list of image files in the directory
    image_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                   if f.lower().endswith(tuple(image_ext))]

    # Process images in parallel with a progress bar and set chunksize
    results = process_map(
        check_and_delete,
        image_files,
        desc="Processing images",
        chunksize=100,  # One image per process
        max_workers=os.cpu_count()//2  # Limit to number of CPU cores
    )

    # Calculate the number of deleted images
    num_deleted = sum(results)

    # Print summary
    print(f"Processed {len(image_files)} images, deleted {num_deleted}.")