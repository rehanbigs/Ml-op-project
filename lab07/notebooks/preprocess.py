import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps

# --- Constants ---
MODEL_MAX_SHAPE = (576, 640)  # (height, width)

def normalize_lighting(image: Image.Image) -> Image.Image:
    """
    Normalizes lighting by subtracting blurred background and removing noise.
    """
    img_array = np.array(image, dtype=np.uint8)
    background = cv2.medianBlur(img_array, 101)
    img_float = img_array.astype(np.float32)
    background_float = background.astype(np.float32)
    normalized_float = img_float - background_float + 128.0
    normalized_float = np.clip(normalized_float, 0, 255)
    normalized = normalized_float.astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(normalized, h=10, templateWindowSize=7, searchWindowSize=21)
    print("    - Lighting and shadows have been normalized (Scanned).")
    return Image.fromarray(denoised)

def extract_text_with_variations(image: Image.Image) -> Image.Image:
    """
    Forces background to black while preserving subtle text variations.
    """
    img_array = np.array(image, dtype=np.uint8)
    optimal_threshold, _ = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = img_array > optimal_threshold
    background_pixels = img_array[~text_mask]
    background_color = np.median(background_pixels) if background_pixels.size > 0 else optimal_threshold
    output_array = np.zeros_like(img_array, dtype=np.float32)
    new_text_values = img_array[text_mask].astype(np.float32) - background_color
    output_array[text_mask] = new_text_values
    final_array = np.clip(output_array, 0, 255).astype(np.uint8)
    print("    - Background forced to pure black, text variations preserved.")
    return Image.fromarray(final_array)

def enhance_text_clarity(image: Image.Image) -> Image.Image:
    """
    Enhances clarity by gentle sharpening and local contrast adjustment.
    """
    img_array = np.array(image, dtype=np.uint8)
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)

    sharpen_kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3, -0.5],
        [0, -0.5, 0]
    ])
    sharpened = cv2.filter2D(closed, -1, sharpen_kernel)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(sharpened)

    smoothed = cv2.fastNlMeansDenoising(contrast, h=5)
    print("    - Text has been sharpened and contrast enhanced for maximum clarity.")
    return Image.fromarray(smoothed)

def trim_borders(image: Image.Image, threshold: int = 245) -> Image.Image:
    """
    Trims white/noisy borders from the image.
    """
    img = np.array(image)
    mask = img < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    trimmed = img[y0:y1, x0:x1]
    return Image.fromarray(trimmed)

def preprocess_image_for_iam_paragraph_model(input_path: Path, output_path: Path) -> None:
    """
    Full Scan-Clean-Enhance preprocessing pipeline.
    """
    print(f"Processing image: {input_path}")
    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    img = img.convert("L")
    img = ImageOps.invert(img)
    print("\n--- Step 1: Image Inverted ---")

    print("\n--- Step 2: Applying Lighting Normalization (Scanning) ---")
    img = normalize_lighting(img)

    print("\n--- Step 3: Extracting Text and Forcing Background to Black ---")
    img = extract_text_with_variations(img)

    print("\n--- Step 4: Enhancing Final Text Clarity ---")
    img = enhance_text_clarity(img)

    print("\n--- Step 5: Trimming white borders ---")
    img = trim_borders(img)

    print(f"\n--- Step 6: Resizing image to fit within {MODEL_MAX_SHAPE[1]}x{MODEL_MAX_SHAPE[0]} ---")
    img_resized = img.copy()
    img_resized.thumbnail((MODEL_MAX_SHAPE[1], MODEL_MAX_SHAPE[0]), Image.Resampling.LANCZOS)
    processed_img = img_resized

    print("\n--- Final Image State ---")
    print(f"    - Final Image size: {processed_img.size}")
    print("    - Image is now clean, clear, smooth, and high-contrast.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_img.save(output_path)
        print(f"\n✅ Successfully processed and saved image to: {output_path}")
    except Exception as e:
        print(f"Error: Could not save the processed image. Reason: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess an image with scanning, cleaning, and enhancement.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input PNG image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed PNG image.")
    args = parser.parse_args()

    input_p = Path(args.input_path)
    output_p = Path(args.output_path)

    preprocess_image_for_iam_paragraph_model(input_path=input_p, output_path=output_p)
