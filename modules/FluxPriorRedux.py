
from diffusers import FluxPriorReduxPipeline
from accelerate.hooks import remove_hook_from_submodules

from .helpers import max_memory, flush, debug
from PIL import Image
import torch

class FluxPriorRedux:
    def __init__(self, model_id="black-forest-labs/FLUX.1-Redux-dev", run_gpu=True):
        self.pipeline = None

        self.model_id = model_id
        self.run_gpu  = run_gpu

        self.dtype  = torch.bfloat16 if run_gpu else torch.float32

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.unload_pipeline()

    def setup(self):
        pipeline_kwargs = {
            "pretrained_model_name_or_path": self.model_id,
            "low_cpu_mem_usage": True,
            "torch_dtype": self.dtype,
        }
        if self.run_gpu:
            pipeline_kwargs["device_map"] = "balanced"
            pipeline_kwargs["max_memory"] = max_memory(scale=0.95)
            self.pipeline = FluxPriorReduxPipeline.from_pretrained(**pipeline_kwargs)
        else:
            self.pipeline = FluxPriorReduxPipeline.from_pretrained(**pipeline_kwargs).to("cpu")    
               

    def get_embeddings(self, init_image: Image.Image = None):
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not initialized.")
        
        with torch.no_grad():
            pipe_prior_output = self.pipeline(init_image)

        prompt_embeds = pipe_prior_output.prompt_embeds.detach().cpu().detach()
        pooled_prompt_embeds = pipe_prior_output.pooled_prompt_embeds.cpu().detach()
       
        return prompt_embeds, pooled_prompt_embeds
    
    def unload_pipeline(self):
        try:
            if self.pipeline and hasattr(self.pipeline, "reset_device_map"):
                self.pipeline.reset_device_map()
                self.remove_accelerate_hooks()

            if self.pipeline:
                del self.pipeline.image_encoder
                del self.pipeline
                self.pipeline = None

            flush(self.__class__.__name__, 0.95) 
            debug(self.pipeline, "unload_pipeline")
        except Exception as error:
            raise RuntimeError(f"[FATAL] {error}")


    def remove_accelerate_hooks(self):
        if self.pipeline and hasattr(self.pipeline, "_hf_hook"):
            remove_hook_from_submodules(self.pipeline)
