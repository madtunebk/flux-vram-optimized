from diffusers import FluxImg2ImgPipeline, FluxPipeline, AutoModel, AutoencoderKL

from PIL import Image
from accelerate.hooks import remove_hook_from_submodules

from .helpers import flush, debug

import torch

class ImageGenerator:
    def __init__(self, model_id:str = "black-forest-labs/FLUX.1-dev", max_memory:dict = {}, torch_dtype:torch.dtype = torch.bfloat16, full_load:bool = False, img2mg:bool = False):
        self.pipeline     = None
        self.transformer  = None
        self.transformer_8bit = None
        self.vae_encoder  = None

        self.model_id     = model_id
        self.max_memory   = max_memory
        self.torch_dtype  = torch_dtype
        self.full_load    = full_load
        self.img2mg       = img2mg

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unload_pipeline()

    def setup(self):
        """
        Initialize the diffusion pipeline. If fullload is True,
        load a separate transformer and inject it into pipeline kwargs.
        """
        self.pipeline_cls = FluxImg2ImgPipeline if self.img2mg else FluxPipeline

        pipeline_kwargs = {
            "pretrained_model_name_or_path" : self.model_id,
            "text_encoder" : None,
            "text_encoder_2" : None,
            "tokenizer": None,
            "tokenizer_2": None,
            "vae" : None,
            "torch_dtype": self.torch_dtype 
        }
        if self.full_load:
            transformer_kwargs = {
                "pretrained_model_or_path" : self.model_id,
                "subfolder" : "transformer",
                "device_map": "balanced",
                "max_memory" : self.max_memory,
                "offload_folder" :"transformer",
                "torch_dtype": self.torch_dtype,
            } 

            # Load transformer separately for full model
            self.transformer = AutoModel.from_pretrained(**transformer_kwargs)

            pipeline_kwargs["transformer"] = self.transformer #self.transformer_8bit
        else:
            # Required for GPU loading when not injecting transformer
            pipeline_kwargs['device_map']  = "balanced"
            pipeline_kwargs['max_memory']  = self.max_memory
            pipeline_kwargs['torch_dtype'] = self.torch_dtype 
        
        if self.img2mg:
            vae_kwargs = {
                "pretrained_model_name_or_path": self.model_id,
                "subfolder": "vae",
                "device_map": "balanced",
                "max_memory" : self.max_memory,
                "torch_dtype": self.torch_dtype
            }
            self.vae_encoder  = AutoencoderKL.from_pretrained(**vae_kwargs)
            pipeline_kwargs["vae"] = self.vae_encoder 


        self.pipeline = self.pipeline_cls.from_pretrained(**pipeline_kwargs)

        self.pipeline.enable_attention_slicing()
        debug(self.pipeline, "ImageGenerator")  

    def run_generator(
        self,
        init_image: Image.Image = None,
        prompt_embeds: torch.Tensor = None,
        pooled_prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds = None,
        negative_pooled_prompt_embeds = None,
        width: int = 1024,
        height: int = 1024,
        num_inference: int = 8,
        strength: float = 0.90,
        cfg: float = 3.5,
        seed: int = 0,
    ):
        # Validate required inputs
        if prompt_embeds is None or pooled_prompt_embeds is None:
            raise RuntimeError("[FATAL] Missing prompt embeddings")
        
        # Assemble pipeline kwargs
        pipeline_kwargs = {
            "prompt_embeds": ensure_dtype(prompt_embeds, self.torch_dtype),
            "pooled_prompt_embeds": ensure_dtype(pooled_prompt_embeds, self.torch_dtype),
            "negative_prompt_embeds": ensure_dtype(negative_prompt_embeds, self.torch_dtype),#.to(device),
            "negative_pooled_prompt_embeds": ensure_dtype(negative_pooled_prompt_embeds, self.torch_dtype),#.to(device),
            "width": width,
            "height": height,
            "num_inference_steps": num_inference,
            "guidance_scale": cfg,
            "generator": seed,
            "output_type": "latent",
        }
        if self.img2mg:
            if init_image is None:
                raise RuntimeError("[FATAL] init_image is required for img2img generation")
            
            pipeline_kwargs["image"] = init_image
            pipeline_kwargs["strength"] = strength

        try:
            with torch.no_grad():
                result = self.pipeline(**pipeline_kwargs)

            return result.images
        except Exception as error:
            raise RuntimeError(F"[FATAL]: {error}")

    def status(self):
        if self.pipeline is None:
            print(f"Initializing model. Full load: {self.full_load}")
            self.setup()
            return True
        return False
 
    def unload_pipeline(self): 
        try: 
            self.pipeline.reset_device_map()
            self.remove_accelerate_hooks()

            if self.transformer:              
               del self.transformer
               self.transformer = None 

            if self.vae_encoder:
               del self.vae_encoder
               self.vae_encoder = None

            if self.pipeline:
                del self.pipeline
                self.pipeline = None

            flush(self.__class__.__name__, 0.95)
        except Exception as error:
            raise RuntimeError(f"[FATAL] {error}")       

    def remove_accelerate_hooks(self):
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, "_hf_hook"):
                print(attr_name)
                remove_hook_from_submodules(attr_value)

        if self.pipeline:
            for attr_name, attr_value in self.pipeline.__dict__.items():
                if hasattr(attr_value, "_hf_hook"):
                    remove_hook_from_submodules(attr_value)
