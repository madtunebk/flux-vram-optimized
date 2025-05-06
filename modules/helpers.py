import torch 
import gc 

from time import sleep as delay

def unpack_laten(images_latens, height: int, width: int, scale_factor:int) -> torch.Tensor:
    batch_size, num_patches, channels = images_latens.shape

    height = 2 * (int(height) // (scale_factor * 2))
    width  = 2 * (int(width) // (scale_factor * 2))

    view_latents = images_latens.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    permute_latents = view_latents.permute(0, 3, 1, 4, 2, 5)
    reshape_latents = permute_latents.reshape(batch_size, channels // (2 * 2), height, width)

    return reshape_latents

def ensure_dtype(tensor, dtype):
    if tensor is None:
        return None
    return tensor if tensor.dtype == dtype else tensor.to(dtype=dtype)

def max_memory(scale: float = 0.97) -> dict:
    map_devices = {}
    for i in range(torch.cuda.device_count()):
        total_bytes = torch.cuda.get_device_properties(i).total_memory
        map_devices[i] = int(total_bytes * scale)  # ← in bytes
    return map_devices

def flush(task: str = "Default", wait_time: float = 0.5):
    print(f"\n🧨 Flushing resources: {task}...")
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # <-- VERY useful in multi-process setups
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print(f"⚡ GPU cache cleared and synchronized. {task}.")
        delay(wait_time)
    except RuntimeError as e:
        print(f"⚠️ flush() failed (likely no CUDA context): {e}")
        delay(wait_time)

def debug(pipeline, module_name:str) -> None:
    if pipeline is not None:
        print(f"[{module_name}] Loaded modules ...")
        for key, value in zip(pipeline.config.keys(), pipeline.config.values()):
            if isinstance(value, tuple) and value[0] is not None:
                print(f"↳ {key} : {value[1]}")
                

def fix_size(size: int, lower: int = 200, upper: int = 1920, multiple: int = 16) -> int:
    if not isinstance(size, int):
        raise ValueError(f"Size must be an integer. Got: {type(size).__name__}")
    if not lower <= size <= upper:
        raise ValueError(f"Size must be between {lower} and {upper}. Got: {size}")
    return (size // multiple) * multiple
