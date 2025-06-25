"""Provide an image of handwritten text and get back out a string!"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable, List, Tuple
import tempfile

import gradio as gr
from PIL import Image
from PIL import ImageStat
import requests

from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent
FAVICON = APP_DIR / "1f95e.png"
README = APP_DIR / "README.md"

DEFAULT_PORT = 11700


class PredictorBackend:
    """
    Interface to a backend that serves predictions.
    It can now handle both single paragraph prediction and full form processing,
    returning a downloadable JSON for form results.
    """

    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint_single
        else:
            self.model = ParagraphTextRecognizer()

    def run(self, image: Image, process_as_form: bool) -> str:
        """
        Runs prediction based on the mode selected in the UI.

        Args:
            image: The input PIL Image.
            process_as_form: A boolean from the Gradio checkbox.

        Returns:
            - If not a form: A string with the prediction.
            - If a form: The file path to a temporary JSON file with the results.
        """
        if image is None:
            return "Please provide an image."

        if process_as_form:
            # --- FORM PROCESSING MODE ---
            logging.info("Starting form processing mode...")
            predictions: List[Tuple[int, str]] = self.model.predict_on_form(image, display_cells=False)
            
            if not predictions:
                return "No cells were detected on the form."
            
            # --- JSON CREATION ---
            # Create a dictionary in the desired format
            prediction_dict = {f"Cell {index}": text for index, text in predictions}
            
            # Create a temporary file to save the JSON
            # Using NamedTemporaryFile with delete=False so Gradio can access it.
            # Gradio will handle cleaning up the temp file.
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", encoding='utf-8') as tmp_json:
                json.dump(prediction_dict, tmp_json, indent=4)
                # Return the path to the created JSON file
                return tmp_json.name
        else:
            # --- SINGLE PREDICTION MODE ---
            logging.info("Starting single prediction mode...")
            pred, metrics = self._predict_with_metrics_single(image)
            self._log_inference(pred, metrics)
            # The File component can also display plain text, so this works.
            return pred

    # --- Methods for Single Prediction ---
    def _predict_with_metrics_single(self, image):
        pred = self.model.predict(image, display_image=False)
        stats = ImageStat.Stat(image.convert("L")) # Use grayscale for stats
        metrics = {
            "image_mean_intensity": stats.mean,
            "image_median": stats.median,
            "image_extrema": stats.extrema,
            "image_area": image.size[0] * image.size[1],
            "pred_length": len(pred),
        }
        return pred, metrics

    def _predict_from_endpoint_single(self, image):
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


def make_frontend(fn: Callable[[Image, bool], str]):
    """Creates a gradio.Interface frontend for the prediction function."""
    examples_dir = Path("text_recognizer") / "tests" / "support" / "paragraphs"
    form_example = Path("data/raw/image1.jpeg") 
    
    example_fnames = [elem for elem in os.listdir(examples_dir) if elem.endswith(".png")]
    example_paths = [examples_dir / fname for fname in example_fnames]
    
    if form_example.exists():
        example_paths.append(form_example)
        
    examples = [[str(path)] for path in example_paths]
    
    readme = _load_readme()

    # Build the interface
    frontend = gr.Interface(
        fn=fn,
        # --- INPUTS ---
        inputs=[
            gr.components.Image(type="pil", label="Handwritten Text Document"),
            gr.components.Checkbox(label="Process as Full Form", value=False)
        ],
        # --- OUTPUT ---
        # Changed to a File component to handle JSON download
        outputs=gr.components.File(label="Prediction Result"),
        
        title="📝 Text Recognizer & Form Extractor",
        description="Provide an image of handwritten text. For single lines/paragraphs, get back text. For forms, check the box to get a downloadable JSON of all cell contents.",
        article=readme,
        examples=examples,
        cache_examples=False,
        allow_flagging="never",
        favicon_path=FAVICON,
    )

    return frontend


def _load_readme(with_logging=False):
    readme_path = APP_DIR / "README.md"
    if not readme_path.exists():
        return ""
    with open(readme_path) as f:
        lines = f.readlines()
        if not with_logging:
            try:
                stop_index = lines.index("<!-- logging content below -->\n")
                lines = lines[:stop_index]
            except ValueError:
                pass
        readme = "".join(lines)
    return readme


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_url", default=None, type=str)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
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