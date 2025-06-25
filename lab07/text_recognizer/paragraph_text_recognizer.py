"""
Detects text in paragraphs or in all cells of a form image.
"""
import argparse
import logging
from pathlib import Path
from typing import Sequence, Union, List, Tuple

from PIL import Image, ImageOps
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from text_recognizer import util
from text_recognizer.stems.paragraph import ParagraphStem
# Ensure this filename matches your cell extraction script
from table_extractor import extract_cell_coordinates


STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "paragraph-text-recognizer"
MODEL_FILE = "lunar.pt"

MODEL_MAX_SHAPE = (576, 640)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class ParagraphTextRecognizer:
    """Recognizes text in a single paragraph image or in all cells of a form."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
            print(f"Using default model path: {model_path}")
        self.model = torch.jit.load(model_path)
        self.mapping = self.model.mapping
        self.ignore_tokens = self.model.ignore_tokens
        self.stem = ParagraphStem()

    # --- START: OCR Preprocessing Helper Methods ---
    def _normalize_lighting(self, image: Image.Image) -> Image.Image:
        """Normalizes lighting by subtracting blurred background and removing noise."""
        img_array = np.array(image, dtype=np.uint8)
        background = cv2.medianBlur(img_array, 101)
        img_float = img_array.astype(np.float32)
        background_float = background.astype(np.float32)
        normalized_float = img_float - background_float 
        normalized_float = np.clip(normalized_float, 0, 255)
        normalized = normalized_float.astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(normalized, h=10, templateWindowSize=7, searchWindowSize=21)
        return Image.fromarray(denoised)

    def _extract_text_with_variations(self, image: Image.Image) -> Image.Image:
        """Forces background to black while preserving subtle text variations."""
        img_array = np.array(image, dtype=np.uint8)
        optimal_threshold, _ = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_mask = img_array > optimal_threshold
        background_pixels = img_array[~text_mask]
        background_color = np.median(background_pixels) if background_pixels.size > 0 else optimal_threshold
        output_array = np.zeros_like(img_array, dtype=np.float32)
        new_text_values = img_array[text_mask].astype(np.float32) - background_color
        output_array[text_mask] = new_text_values
        final_array = np.clip(output_array, 0, 255).astype(np.uint8)
        return Image.fromarray(final_array)

    def _enhance_text_clarity(self, image: Image.Image) -> Image.Image:
        """Enhances clarity by gentle sharpening and local contrast adjustment."""
        img_array = np.array(image, dtype=np.uint8)
        kernel = np.ones((2, 2), np.uint8)
        closed = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        sharpen_kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(closed, -1, sharpen_kernel)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(sharpened)
        smoothed = cv2.fastNlMeansDenoising(contrast, h=5)
        return Image.fromarray(smoothed)

    def _trim_borders(self, image: Image.Image, threshold: int = 245) -> Image.Image:
        """Trims white/noisy borders from the image."""
        img_array = np.array(image)
        mask = img_array < threshold
        coords = np.argwhere(mask)
        if coords.size == 0:
            return image
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        trimmed = img_array[y0:y1, x0:x1]
        return Image.fromarray(trimmed)

    def _run_ocr_preprocessing_pipeline(self, image: Image.Image) -> Image.Image:
        """Full Scan-Clean-Enhance preprocessing pipeline for a single image cell."""
        img = image.convert("L")
        img = ImageOps.invert(img)
        img = self._normalize_lighting(img)
        img = self._extract_text_with_variations(img)
        img = self._enhance_text_clarity(img)
        img = self._trim_borders(img)
        img.thumbnail((MODEL_MAX_SHAPE[1], MODEL_MAX_SHAPE[0]), Image.Resampling.LANCZOS)
        return img
        
    # --- CORE PREDICTION METHODS ---
    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image], display_image: bool = True) -> str:
        """Predicts text in a SINGLE image."""
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = util.read_image_pil(image, grayscale=False)
        processed_image_pil = self._run_ocr_preprocessing_pipeline(image_pil)
        if display_image:
            plt.imshow(processed_image_pil, cmap='gray'); plt.title("Preprocessed Cell"); plt.axis('off'); plt.show()
        image_tensor = self.stem(processed_image_pil).unsqueeze(axis=0)
        y_pred = self.model(image_tensor)[0]
        pred_str = convert_y_label_to_string(y=y_pred, mapping=self.mapping, ignore_tokens=self.ignore_tokens)
        return pred_str

    def predict_on_form(self, image: Union[str, Path, Image.Image], display_cells: bool = True) -> List[Tuple[int, str]]:
        """
        Performs full-form processing using the "Detect on Low-Res, Crop on High-Res" method,
        with added padding to exclude cell borders from the final crop.
        """
        print(f"--- Starting Full Form Processing ---")
        
        # --- Step 1: Load the original, high-resolution image ---
        image_pil_high_res = None
        if isinstance(image, (str, Path)):
            try:
                image_pil_high_res = Image.open(image).convert("RGB")
            except FileNotFoundError:
                print(f"Error: Could not load image from path: {image}")
                return []
        elif isinstance(image, Image.Image):
            image_pil_high_res = image.convert("RGB")
        else:
            raise TypeError("Input 'image' must be a file path, a Path object, or a PIL.Image object.")

        # --- Step 2: Detect cells on a low-resolution copy ---
        print("\nStep 2: Detecting table cells on a low-resolution copy...")
        image_np_high_res = cv2.cvtColor(np.array(image_pil_high_res), cv2.COLOR_RGB2BGR)
        low_res_coords, resized_image_for_debug = extract_cell_coordinates(image_np_high_res)

        if not low_res_coords:
            print("No cells were detected. Aborting.")
            return []
        
        # --- Step 3: Scale coordinates back to the high-resolution image size ---
        print("\nStep 3: Scaling coordinates to match original image resolution...")
        high_res_h, high_res_w = image_np_high_res.shape[:2]
        low_res_h, low_res_w = resized_image_for_debug.shape[:2]

        w_scale = high_res_w / low_res_w
        h_scale = high_res_h / low_res_h

        high_res_coords = [
            (int(x * w_scale), int(y * h_scale), int(w * w_scale), int(h * h_scale))
            for (x, y, w, h) in low_res_coords
        ]

        # --- Step 4: Loop, apply padding, crop from HIGH-RES image, and run OCR ---
        print("\nStep 4: Running OCR on padded, high-resolution cell crops...")
        predictions = []
        
        # Define a padding value to shrink the bounding box.
        # This value can be tuned. 3-5 pixels is usually a good start.
        padding = 4 

        for i, (x, y, w, h) in enumerate(high_res_coords):
            # Apply padding to the coordinates to create an inset box
            padded_x = x + padding
            padded_y = y + padding
            padded_w = w - (2 * padding)
            padded_h = h - (2 * padding)
            
            # Ensure the padded box is still valid (has positive width/height)
            if padded_w <= 0 or padded_h <= 0:
                print(f"  Skipping cell {i} because it's too small for padding.")
                continue

            # Define the crop box using the new padded coordinates for PIL: (left, upper, right, lower)
            crop_box = (padded_x, padded_y, padded_x + padded_w, padded_y + padded_h)
            
            # Crop from the original, high-quality PIL image
            cell_crop_pil = image_pil_high_res.crop(crop_box)
            
            # The predict function will now receive a crop without the black border lines
            predicted_text = self.predict(cell_crop_pil, display_image=display_cells)
            
            predictions.append((i, predicted_text))
            
            if not display_cells: print(f"  Cell {i}: '{predicted_text}'")
            else: print(f"Prediction for Cell {i}: '{predicted_text}'")
            
        print("\n--- Full Form Processing Complete ---")
        return predictions


def convert_y_label_to_string(y: torch.Tensor, mapping: Sequence[str], ignore_tokens: Sequence[int]) -> str:
    return "".join([mapping[i] for i in y if i not in ignore_tokens])

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("filename", type=str, help="Path to the image file.")
    parser.add_argument("--form", action="store_true", help="Run in full-form processing mode.")
    parser.add_argument("--display", action="store_true", help="Display preprocessed cell images before OCR.")
    args = parser.parse_args()

    text_recognizer = ParagraphTextRecognizer()

    if args.form:
        all_predictions = text_recognizer.predict_on_form(args.filename, display_cells=args.display)
        print("\n--- Final Predictions Summary ---")
        for index, text in all_predictions:
            print(f"Cell Index {index}: {text}")
    else:
        logging.getLogger().setLevel(logging.WARNING)
        pred_str = text_recognizer.predict(args.filename, display_image=args.display)
        print("\n--- Single Image Prediction ---")
        print(pred_str)


if __name__ == "__main__":
    main()