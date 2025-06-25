#!/usr/bin/env python3
"""
stagelocalckpt.py – A generalized script to convert a local Lightning .ckpt
to a production-ready TorchScript (.pt).

This script intelligently extracts hyperparameters from the checkpoint file,
removing the need for hardcoded model parameters.

Example
-------
# The script will automatically find the model and data classes if saved in the ckpt
python stagelocalckpt.py --ckpt model.ckpt --out model.pt

# You can still override them if needed
python stagelocalckpt.py \
    --ckpt model.ckpt \
    --out  model.pt \
    --data_class CustomParagraphs \
    --model_class BigTransformer
"""

from __future__ import annotations
from pathlib import Path
import argparse
import torch
from text_recognizer.lit_models import TransformerLitModel
from training.util import setup_data_and_model_from_args
import pprint

def _extract_hparams_robust(ckpt_dict: dict) -> dict:
    """
    Intelligently extracts hyperparameters from a checkpoint dictionary.

    It searches in a prioritized order to handle different Lightning versions:
    1. A nested 'args' or 'hparams' dict inside 'hyper_parameters'.
    2. The 'hyper_parameters' dict itself.
    3. The top-level dictionary of the checkpoint.
    """
    # Start with top-level keys as the lowest priority fallback
    hparams = {k: v for k, v in ckpt_dict.items() if isinstance(v, (int, float, str, bool, list, tuple))}

    # Update with 'hyper_parameters' keys (medium priority)
    if "hyper_parameters" in ckpt_dict and isinstance(ckpt_dict["hyper_parameters"], dict):
        hparams.update(ckpt_dict["hyper_parameters"])
        print("  > Found hyperparameters in 'hyper_parameters' dict.")

    # Update with the most specific nested dict (highest priority)
    hp_nested = ckpt_dict.get("hyper_parameters", {})
    for key in ("args", "hparams", "params"):
        if isinstance(hp_nested, dict) and key in hp_nested and isinstance(hp_nested[key], dict):
            hparams.update(hp_nested[key])
            print(f"  > Found most specific hyperparameters in 'hyper_parameters/{key}'.")
            break  # Stop after finding the first, most likely source
            
    if not hparams:
        print("  > WARNING: Could not find any hyperparameters in the checkpoint.")

    return hparams


def _merged_namespace(cli_args: argparse.Namespace, ckpt_hparams: dict) -> argparse.Namespace:
    """
    Merges hyperparameters from the checkpoint with command-line overrides.
    CLI arguments take precedence over checkpoint hyperparameters.
    """
    merged_params = ckpt_hparams.copy()
    
    # Get provided CLI arguments (don't include defaults that were not set)
    cli_override = {k: v for k, v in vars(cli_args).items() if v is not None}
    
    # Let CLI arguments override checkpoint hparams
    merged_params.update(cli_override)

    return argparse.Namespace(**merged_params)


def convert(
    ckpt: Path,
    out: Path,
    data_class: str | None = None,
    model_class: str | None = None,
    strict: bool = False,
):
    """
    Main conversion function.
    """
    ckpt = ckpt.expanduser().resolve()
    out  = out.expanduser().resolve()

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")

    print(f"Loading checkpoint from: {ckpt}")
    ckpt_dict = torch.load(str(ckpt), map_location="cpu")

    print("Extracting hyperparameters...")
    hp = _extract_hparams_robust(ckpt_dict)

    # Combine CLI args and checkpoint hparams
    cli_ns = argparse.Namespace(data_class=data_class, model_class=model_class)
    args_ns = _merged_namespace(cli_ns, hp)
    
    print("\nFinal parameters used to build the model:")
    pprint.pprint(vars(args_ns))

    # --- Validate that essential parameters exist ---
    required_keys = ["model_class", "data_class"]
    for key in required_keys:
        if not hasattr(args_ns, key) or getattr(args_ns, key) is None:
            raise ValueError(
                f"Crucial parameter '{key}' was not found in the checkpoint. "
                f"Please provide it as a command-line argument (e.g., --{key} MyClassName)."
            )

    # Recreate base model and load LightningModule
    print("\nRebuilding model architecture and loading weights...")
    _, base_model = setup_data_and_model_from_args(args_ns)

    lit_model = TransformerLitModel.load_from_checkpoint(
        checkpoint_path=str(ckpt),
        args=args_ns,
        model=base_model,
        strict=strict,
    ).eval()
    print("Model and weights loaded successfully.")

    # Save TorchScript
    print(f"\nSaving TorchScript model to: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(lit_model.to_torchscript(method="script"), str(out))
    print(f"✅ TorchScript model saved successfully!")


# ------------------------------ CLI -----------------------------------------
def cli() -> argparse.Namespace:
    """Defines the command-line interface."""
    p = argparse.ArgumentParser(
        description="A generalized script to convert a local .ckpt to a production TorchScript .pt model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--ckpt", type=Path, required=True, help="Path to the input PyTorch Lightning checkpoint (.ckpt).")
    p.add_argument("--out",  type=Path, default=Path("model.pt"), help="Path to save the output TorchScript model (.pt).")
    p.add_argument("--data_class", type=str, default=None, help="(Optional) Override the data class name. Needed if not in checkpoint.")
    p.add_argument("--model_class", type=str, default=None, help="(Optional) Override the model class name. Needed if not in checkpoint.")
    p.add_argument("--strict", action="store_true", help="Enable strict loading of weights. Defaults to False.")
    return p.parse_args()


if __name__ == "__main__":
    ns = cli()
    convert(
        ckpt=ns.ckpt,
        out=ns.out,
        data_class=ns.data_class,
        model_class=ns.model_class,
        strict=ns.strict,
    )