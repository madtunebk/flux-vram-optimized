import torch, gc
from accelerate.hooks import remove_hook_from_submodules

from .helpers import flush, get_flux1_embeddings
from diffusers import FluxPipeline

class TextEncoder:
    """
    Context manager to load/unload only the CLIP text encoders and tokenizers
    for Flux1 without loading the full pipeline.
    """
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32, device: str = "cpu"):
        self.model_id    = model_id
        self.torch_dtype = torch_dtype
        self.device      = device
        self.pipeline    = None

    def __enter__(self):
        self.setup()
        return self

    def setup(self):
        # Initialize FluxPipeline with only text modules loaded
        self.pipeline = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            vae=None,
            #transformer=None,
            torch_dtype=self.torch_dtype
        ).to(self.device)

        # Extract loaded modules
        self.tokenizer      = self.pipeline.tokenizer
        self.tokenizer_2    = self.pipeline.tokenizer_2
        self.text_encoder   = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2

    def get_embeddings(self, prompt: str, negative_prompt: str = None):
        """
        Returns a tuple (t5_embeddings, clip_embeddings).
        Auto-initializes if not yet set up.
        """
        if self.pipeline is None:
            self.setup()
        # Generate embeddings using standalone utility
        return get_flux1_embeddings(
            text_encoder   = self.text_encoder,
            text_encoder_2 = self.text_encoder_2,
            tokenizer      = self.tokenizer,
            tokenizer_2    = self.tokenizer_2,
            prompt         = prompt,
            negative_prompt= negative_prompt,
            device         = self.device
        )

    def unload_pipeline(self):
        """
        Remove hooks and free GPU memory used by text encoders.
        """
        if self.pipeline:
            # Reset any device mapping and remove hooks
            try:
                self.pipeline.reset_device_map()
            except Exception:
                pass
            for attr in (self.text_encoder, self.text_encoder_2):
                try:
                    remove_hook_from_submodules(attr)
                except Exception:
                    pass
        # Delete references and clear cache
        for attr in ('text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'pipeline'):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()
        flush(self.__class__.__name__, 0.95)

    def __exit__(self, exc_type, exc_value, traceback):
        self.unload_pipeline()
