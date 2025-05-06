"""
TextEncoder Module

# TODO: Warning - Ensure that the CLIP model and tokenizer versions match, and that you have sufficient VRAM to load the text encoder on your device.
"""
from transformers import CLIPTokenizer, CLIPTextModel
import torch

class TextEncoder:
    """
    Encodes text prompts into CLIP-style embeddings for downstream pipelines.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda:0"):
        # TODO: Warn if CUDA device is not available or VRAM is insufficient
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({device}) but torch.cuda.is_available() is False.")
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(self.device)

    def encode(self, text: str) -> torch.Tensor:
        """
        Tokenizes and encodes a text prompt into embeddings.

        Args:
            text (str): The text prompt to encode.

        Returns:
            torch.Tensor: The encoded text embeddings.
        """
        # TODO: Add length checks and warnings for empty or excessively long prompts
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
