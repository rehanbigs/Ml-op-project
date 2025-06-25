"""
A script to run inference using a LineCNNSimple model from a .ckpt file,
or to convert the .ckpt to a TorchScript (.pt) file.
Includes functionality to display the preprocessed image.
"""
import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import OrderedDict

import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt # Added for displaying the image
import numpy as np
import cv2
# --- Constants ---
IMAGE_SIZE = 28
WINDOW_WIDTH = IMAGE_SIZE
WINDOW_STRIDE = IMAGE_SIZE
NUM_CLASSES = 83 # Based on previous error logs

# --- Model Definitions ---

class CNN(nn.Module):
    """
    Corrected CNN architecture based on reverse-engineering the .ckpt file.
    - Conv1 has 64 output channels.
    - Conv2 has 64 input channels.
    - Only ONE pooling layer is used.
    """
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        
        self.num_classes = len(data_config["mapping"])
        input_dims = data_config["input_dims"]
        
        # Corrected Architecture
        self.conv1 = nn.Conv2d(input_dims[0], 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        conv_output_size = self._get_conv_output(input_dims)
        
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, self.num_classes)

    def _get_conv_output(self, shape: Tuple[int, int, int]) -> int:
        with torch.no_grad():
            x = torch.rand(1, *shape)
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.relu2(self.conv2(x))
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        return parser


class LineCNNSimple(nn.Module):
    """LeNet based model that takes a line of characters."""
    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("window_stride", WINDOW_STRIDE)
        self.limit_output_length = self.args.get("limit_output_length", False)

        self.num_classes = len(data_config["mapping"])
        self.output_length = data_config.get("output_dims", (100,))[0]
        cnn_input_dims = (data_config["input_dims"][0], self.WW, self.WW)
        
        cnn_data_config = data_config.copy()
        cnn_data_config["input_dims"] = cnn_input_dims
        self.cnn = CNN(data_config=cnn_data_config, args=args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = x.shape
        assert H == IMAGE_SIZE, f"Input image height must be {IMAGE_SIZE}"

        S = math.floor((W - self.WW) / self.WS + 1)

        activations = torch.zeros((B, self.num_classes, S)).type_as(x)
        for s in range(S):
            start_w = self.WS * s
            end_w = start_w + self.WW
            window = x[:, :, :, start_w:end_w]
            activations[:, :, s] = self.cnn(window)

        if self.limit_output_length:
            activations = activations[:, :, : self.output_length]
        return activations

    @staticmethod
    def add_to_argparse(parser):
        CNN.add_to_argparse(parser)
        parser.add_argument("--window_width", type=int, default=WINDOW_WIDTH)
        parser.add_argument("--window_stride", type=int, default=WINDOW_STRIDE)
        parser.add_argument("--limit_output_length", action="store_true", default=False)
        return parser

# --- Utility and Inference Functions ---

def clean_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """Cleans the keys of a state_dict loaded from a PyTorch Lightning checkpoint."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model."):
            k = k[len("model."):]
        k = k.replace(".conv.weight", ".weight")
        k = k.replace(".conv.bias", ".bias")
        new_state_dict[k] = v
    return new_state_dict

def preprocess_image(image_path: Path) -> torch.Tensor:
    """Load and preprocess an image for the LineCNNSimple model."""
    print("Preprocessing image...")
    image = Image.open(image_path).convert("L")

    w, h = image.size
    new_w = int(w * (IMAGE_SIZE / h))
    image = image.resize((new_w, IMAGE_SIZE), Image.Resampling.LANCZOS)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

def display_processed_image(tensor: torch.Tensor):
    """Converts a preprocessed tensor back to a displayable image and shows it."""
    print("Displaying preprocessed image that will be fed to the model...")
    # Make a copy to avoid changing the original tensor
    tensor_to_show = tensor.clone().detach()
    
    # Remove the batch dimension
    tensor_to_show = tensor_to_show.squeeze(0)
    
    # Reverse the normalization: (tensor * std) + mean
    # Our std and mean were both 0.5
    tensor_to_show = tensor_to_show * 0.5 + 0.5
    
    # Convert tensor to numpy array for displaying
    # The channel is first, so we need to move it to the end: (C, H, W) -> (H, W, C)
    numpy_image = tensor_to_show.permute(1, 2, 0).numpy()
    
    # Squeeze the channel dimension if it's 1 (for grayscale)
    numpy_image = np.squeeze(numpy_image, axis=2)
    
    plt.imshow(numpy_image, cmap='gray')
    plt.title("Preprocessed Image (as seen by model)")
    plt.axis('off')
    plt.show()

def decode_prediction(prediction_tensor: torch.Tensor, mapping: List[str]) -> str:
    """Decodes the model's output tensor into a string."""
    prediction_tensor = prediction_tensor.squeeze(0)
    predicted_indices = torch.argmax(prediction_tensor, dim=0)
    predicted_text = "".join([mapping[i] for i in predicted_indices])
    return predicted_text

def convert_to_jit(model: nn.Module, output_path: Path, dummy_input: torch.Tensor):
    """Traces the model and saves it as a TorchScript file."""
    print(f"Converting model to TorchScript (.pt) at {output_path}...")
    model.eval()
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(output_path)
    print("✅ Model successfully converted to TorchScript.")

def main():
    parser = argparse.ArgumentParser(description="Inference script for LineCNNSimple.")
    parser.add_argument("ckpt_path", type=Path, help="Path to the .ckpt checkpoint file.")
    parser.add_argument("--image_path", type=Path, help="Path to the input image for prediction.")
    parser.add_argument("--convert_to_jit", type=Path, metavar="OUTPUT_PATH", default=None)
    parser.add_argument("--no_display", action="store_true", help="Do not display the preprocessed image.")
    args = parser.parse_args()

    print(f"Loading model from checkpoint: {args.ckpt_path}...")
    
    placeholder_mapping = "".join([chr(i) for i in range(NUM_CLASSES)])
    data_config = {
        "mapping": placeholder_mapping,
        "input_dims": (1, IMAGE_SIZE, IMAGE_SIZE),
        "output_dims": (100,)
    }
    
    model = LineCNNSimple(data_config)
    
    state_dict = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    cleaned_sd = clean_state_dict(state_dict)
    
    model.load_state_dict(cleaned_sd)
    print("Model weights loaded successfully.")
    
    model.eval()

    if args.convert_to_jit:
        dummy_input = torch.randn(1, 1, IMAGE_SIZE, 300)
        convert_to_jit(model, args.convert_to_jit, dummy_input)
        return

    if not args.image_path:
        print("Error: --image_path is required for prediction.")
        return

    if not args.image_path.exists():
        print(f"Error: Image file not found at {args.image_path}")
        return

    image_tensor = preprocess_image(args.image_path)
    
    # Display the image unless suppressed
    if not args.no_display:
        display_processed_image(image_tensor)
    
    print("Running prediction...")
    with torch.no_grad():
        output = model(image_tensor)

    mapping = model.data_config.get("mapping", placeholder_mapping)
    predicted_text = decode_prediction(output, mapping)
    
    print("\n" + "="*30)
    print(f"  Prediction: {predicted_text}")
    print("="*30)

if __name__ == "__main__":
    main()