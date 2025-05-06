import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from accelerate.hooks import remove_hook_from_submodules

from .helpers import unpack_laten, max_memory, flush

class ImageDecoder:
    def __init__(self, model_id="black-forest-labs/FLUX.1-dev", rungpu=True):
        self.model_id = model_id
        self.rungpu = rungpu

        self.vae = None
        self.device = "cuda" if rungpu else "cpu"
        self.dtype = torch.float32  # required for tiled_decode()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unload_pipeline()

    def setup(self):
        vae_kwargs = {
            "pretrained_model_name_or_path": self.model_id,
            "subfolder": "vae",
            "torch_dtype": self.dtype,
        }

        if self.rungpu:
            vae_kwargs["device_map"] = "balanced"
            vae_kwargs["max_memory"] = max_memory()
            self.vae = AutoencoderKL.from_pretrained(**vae_kwargs)
        else:
            self.vae = AutoencoderKL.from_pretrained(**vae_kwargs).to(self.device)

    def decode(self, latents: torch.Tensor, height: int, width: int):
        if self.vae is None:
            raise RuntimeError("VAE not initialized. Call setup() first.")

        self.vae.enable_slicing()
        self.vae.enable_tiling()

        scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=scale_factor)

        latents = latents.to(self.device, dtype=self.dtype)
        latents = unpack_laten(latents, height, width, scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        with torch.no_grad():
            images = self.vae.tiled_decode(latents, return_dict=False)[0]
            return image_processor.postprocess(images, output_type="pil")

    def unload_pipeline(self):
        try:
            if self.vae and hasattr(self.vae, "reset_device_map"):
                self.vae.reset_device_map()

            self.remove_accelerate_hooks()

            if self.vae:
                del self.vae
                self.vae = None

            flush(self.__class__.__name__, 0.95)
        except Exception as error:
            raise RuntimeError(f"[FATAL] {error}")

    def remove_accelerate_hooks(self):
        if self.vae and hasattr(self.vae, "_hf_hook"):
            remove_hook_from_submodules(self.vae)
