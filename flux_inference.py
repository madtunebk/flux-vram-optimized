### Flux Image Generation Script
# This script is structured into three modular pipelines to optimize VRAM usage on consumer GPUs:
#
# 1. **TextEncoder** (`black-forest-labs/FLUX.1-dev` / `black-forest-labs/FLUX.1-schnell`)
#    - converts your text prompt into embeddings via a CLIP-style encoder
#
# 2. **FluxPriorRedux** (`black-forest-labs/FLUX.1-Redux-dev`)
#    - Computes image-conditioned latent prompt embeddings using the Flux Redux architecture.
#
# 3. **ImageGenerator** (`black-forest-labs/FLUX.1-dev` / `black-forest-labs/FLUX.1-schnell`)
#    - Loads minimal text-to-image or image-to-image pipelines (transformers + VAE + FluxImg2ImgPipeline or FluxPipeline).
#    - Offers an optional Redux variation mode for stylized variations of the input image.
#
# 4. **ImageDecoder** (`black-forest-labs/FLUX.1-dev` / `black-forest-labs/FLUX.1-schnell`)
#    - Uses the AutoencoderKL VAE decoder to convert latents back into Pillow `Image` objects.
#
# **Design Rationale:**
# - Modular separation ensures only required components are loaded at each stage, reducing peak VRAM usage.
# - Successfully tested on GPUs with 12 GB VRAM; at least 16 GB VRAM is recommended to avoid OOM errors.
# - Requires at least 2 GPUs to run the full pipeline via `accelerate` with `device_map`.
# - Utilizes `accelerate` with `device_map` to distribute workloads across multiple CUDA devices.

# --- Setup imports and helper functions ---
from modules.TextEncoder import TextEncoder
from modules.FluxPriorRedux import FluxPriorRedux
from modules.ImageGenerator import ImageGenerator
from modules.ImageDecoder import ImageDecoder

import random
from datetime import datetime
from diffusers.utils import load_image

from modules.helpers import max_memory, fix_size, debug
import torch

# --- Model selection ---
# You can switch between different Flux models here
model = "black-forest-labs/FLUX.1-schnell" #  Alternatives:  "ostris/Flex.1-alpha"  "black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"

prompt =  "Ultra High Resolution Art of Superman and Wonder Woman in a thunderous anime-style battle among shattered skyscrapers and storm-warped clouds. Wonder Woman’s armor is cracked, her blade clashing against Superman’s heat vision blast. Floating debris circles them as the sky rains violet lightning. Concept art with vivid brushwork and glowing edge highlights, channeling energy from anime battle epics, award-winning, epic composition, ultra detailed"

# --- Load and prepare the initial image ---
init_image = load_image(
    "temp/flux_20250505_183815.png"
)
# Ensure dimensions are multiples of the model's required size
width, height = fix_size(init_image.width), fix_size(init_image.height)

# --- Generate or load prompt embeddings (tokenizer/text_encoder)---
enable_textencoder = True
if enable_textencoder:
    with TextEncoder() as text_encoder:
        prompt_embeds, pooled_prompt_embeds = text_encoder.get_embeddings(prompt)
        # Save embeddings for debugging or reuse
        torch.save(prompt_embeds, "temp/debug_latents/prompt_embeds.pt")
        torch.save(pooled_prompt_embeds, "temp/debug_latents/pooled_prompt_embeds.pt")      
  
# --- Generate or load prompt embeddings (REDUX)---
enable_redux = False
if enable_redux:
    # Use FluxPriorRedux to compute prompt embeddings from the initial image
    with FluxPriorRedux(run_gpu=False) as Redux:
        debug(Redux.pipeline, "Redux Pipeline Initialized")
        prompt_embeds, pooled_prompt_embeds = Redux.get_embeddings(init_image)
    # Save embeddings for debugging or reuse
    torch.save(prompt_embeds, "temp/debug_latents/prompt_embeds.pt")
    torch.save(pooled_prompt_embeds, "temp/debug_latents/pooled_prompt_embeds.pt")
#else:
#    # Load precomputed embeddings to skip the embedding step
#    prompt_embeds = torch.load("temp/debug_latents/prompt_embeds.pt")
#    pooled_prompt_embeds = torch.load("temp/debug_latents/pooled_prompt_embeds.pt")

# --- Latent generation with ImageGenerator ---
latentgen = True
if latentgen:
    # Reserve a fraction of GPU memory for the model and GUI
    # Recommended scale: 0.90–0.95 (reserve ~1 GB VRAM for OS/GUI tasks)
    maxmem = max_memory(scale=0.90)
    print(f"🌱 Initialized GPU with memory scale: {maxmem}")

    # Initialize the image generator pipeline
    with ImageGenerator(
        model_id=model,
        max_memory=maxmem,
        full_load=True,
        img2mg=False,
        torch_dtype=torch.bfloat16
    ) as latent:
        # Seed the random generator for reproducibility
        seed = random.randint(1e11, 1e12 - 1)
        generator = torch.Generator(device="cuda:1").manual_seed(seed)
        print(f"🌱 Generator initialized with seed: {generator.initial_seed()}")

        # Run the latent generator to produce image latents
        results = latent.run_generator(
            init_image=init_image,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            width=width,
            height=height,
            num_inference=35,
            seed=generator
        )
        # Save the generated latents for debugging or reuse
        torch.save(results, "temp/debug_latents/results.pt")

# --- Decode latents back into images ---
with ImageDecoder(model_id=model, rungpu=True) as decoder:
    # Create a timestamp for the output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the saved latents
    latent = torch.load("temp/debug_latents/results.pt")

    # Decode the latent tensors into actual images
    image = decoder.decode(latents=latent, height=height, width=width) 

    # Save the first generated image with a timestamped filename
    image[0].save(f"temp/flux_{timestamp}.png")
    print(f"📸 Saved image as temp/flux_{timestamp}.png")
