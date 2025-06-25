"""Provide an image of handwritten text and get back out a string!"""
import argparse
import logging
import os
from pathlib import Path
from typing import Callable, Tuple

# Dependencies
import gradio as gr
from PIL import Image, ImageOps, ImageStat
import requests
import numpy as np
import cv2

# Local application imports
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

# --- Constants ---
MODEL_MAX_SHAPE = (576, 640)  # (height, width)
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
APP_DIR = Path(__file__).resolve().parent
FAVICON = APP_DIR / "1f95e.png"
README = APP_DIR / "README.md"
DEFAULT_PORT = 11700


class PredictorBackend:
    """
    Interface to a backend that serves predictions.
    It now includes the full in-memory preprocessing pipeline and returns
    both the processed image and the prediction.
    """
    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = ParagraphTextRecognizer()
            self._predict = model.predict

    # --- In-Memory Preprocessing Pipeline (Copied Verbatim Logic) ---

    def _normalize_lighting(self, image: Image.Image) -> Image.Image:
        """Normalizes lighting by subtracting blurred background and removing noise."""
        img_array = np.array(image, dtype=np.uint8)
        background = cv2.medianBlur(img_array, 101)
        img_float = img_array.astype(np.float32)
        background_float = background.astype(np.float32)
        normalized_float = img_float - background_float + 128.0
        normalized_float = np.clip(normalized_float, 0, 255)
        normalized = normalized_float.astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(normalized, h=10, templateWindowSize=7, searchWindowSize=21)
        logging.info("    - Lighting and shadows have been normalized (Scanned).")
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
        logging.info("    - Background forced to pure black, text variations preserved.")
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
        logging.info("    - Text has been sharpened and contrast enhanced for maximum clarity.")
        return Image.fromarray(smoothed)

    def _trim_borders(self, image: Image.Image, threshold: int = 245) -> Image.Image:
        """Trims white/noisy borders from the image."""
        img_array = np.array(image)
        mask = img_array < threshold
        coords = np.argwhere(mask)
        if coords.size == 0:
            logging.warning("    - Could not trim borders, image may be blank.")
            return image
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        trimmed = img_array[y0:y1, x0:x1]
        return Image.fromarray(trimmed)

    def _run_preprocessing_pipeline(self, image: Image.Image) -> Image.Image:
        """Full Scan-Clean-Enhance preprocessing pipeline, operating in-memory."""
        logging.info("--- Starting Preprocessing Pipeline ---")
        
        # Original script logic:
        img = image.convert("L")
        
        logging.info("\n--- Step 1: Image Inverted ---")
        img = ImageOps.invert(img)

        logging.info("\n--- Step 2: Applying Lighting Normalization (Scanning) ---")
        img = self._normalize_lighting(img)

        logging.info("\n--- Step 3: Extracting Text and Forcing Background to Black ---")
        img = self._extract_text_with_variations(img)

        logging.info("\n--- Step 4: Enhancing Final Text Clarity ---")
        img = self._enhance_text_clarity(img)

        logging.info("\n--- Step 5: Trimming white borders ---")
        img = self._trim_borders(img)

        logging.info(f"\n--- Step 6: Resizing image to fit within {MODEL_MAX_SHAPE[1]}x{MODEL_MAX_SHAPE[0]} ---")
        # Use a copy for resizing to avoid issues with some Pillow versions
        img_resized = img.copy()
        img_resized.thumbnail((MODEL_MAX_SHAPE[1], MODEL_MAX_SHAPE[0]), Image.Resampling.LANCZOS)
        
        logging.info("\n--- Final Image State ---")
        logging.info(f"    - Final Image size: {img_resized.size}")
        logging.info("    - Image is now clean, clear, smooth, and high-contrast.")
        logging.info("--- Preprocessing Pipeline Finished ---")
        return img_resized

    # --- Main Prediction Logic ---

    def run(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """
        Takes a PIL image, runs it through the full in-memory preprocessing pipeline,
        and then returns the processed image and the model's prediction.
        """
        if image is None:
            # Create a blank image to return as a placeholder
            blank_image = Image.new("L", (200, 50), color=255)
            # Add text to the blank image to inform the user
            from PIL import ImageDraw
            draw = ImageDraw.Draw(blank_image)
            draw.text((10, 10), "Please provide an image.", fill=0)
            return blank_image, "No image provided."

        # 1. Run the full preprocessing pipeline on the input image.
        processed_image = self._run_preprocessing_pipeline(image)

        # 2. Continue with prediction using the preprocessed image.
        pred, metrics = self._predict_with_metrics(processed_image)
        self._log_inference(pred, metrics)
        
        # 3. Return both the processed image and the prediction.
        return processed_image, pred

    def _predict_with_metrics(self, image: Image.Image):
        pred = self._predict(image)
        stats = ImageStat.Stat(image.convert("L")) # Ensure grayscale for stats
        metrics = {
            "image_mean_intensity": stats.mean,
            "image_median": stats.median,
            "image_extrema": stats.extrema,
            "image_area": image.size[0] * image.size[1],
            "pred_length": len(pred),
        }
        return pred, metrics

    def _predict_from_endpoint(self, image: Image.Image):
        encoded = util.encode_b64_image(image)
        payload = {"image": encoded}
        r = requests.post(self.url, json=payload, timeout=15)
        r.raise_for_status()
        pred = r.json()["pred"]
        return pred

    def _log_inference(self, pred, metrics):
        for key, value in metrics.items():
            logging.info(f"METRIC {key} {value}")
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


def make_frontend(fn: Callable[[Image.Image], Tuple[Image.Image, str]]):
    """Creates a gradio.Interface frontend for an image to text function."""
    examples_dir = Path("text_recognizer") / "tests" / "support" / "paragraphs"
    example_fnames = [elem for elem in os.listdir(examples_dir) if elem.endswith(".png")]
    example_paths = [examples_dir / fname for fname in example_fnames]

    examples = [[str(path)] for path in example_paths]
    allow_flagging = "never"
    readme = _load_readme(with_logging=allow_flagging == "manual")

    return gr.Interface(
        fn=fn,
        # Define two outputs: one for the processed image, one for the text
        outputs=[
            gr.components.Image(type="pil", label="Processed Image"),
            gr.components.Textbox(label="Recognized Text")
        ],
        inputs=gr.components.Image(type="pil", label="Handwritten Text"),
        title="📝 Text Recognizer",
        thumbnail=FAVICON,
        description=__doc__,
        article=readme,
        examples=examples,
        cache_examples=False,
        allow_flagging=allow_flagging,
    )


def _load_readme(with_logging=False):
    with open(README) as f:
        lines = f.readlines()
        if not with_logging:
            lines = lines[: lines.index("<!-- logging content below -->\n")]
        readme = "".join(lines)
    return readme


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data...",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )
    return parser


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(predictor.run)
    frontend.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=True,
        favicon_path=FAVICON,
    )


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)