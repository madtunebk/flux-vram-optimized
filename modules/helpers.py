import torch 
import gc 
import re

from collections import deque
from typing import List, Tuple, Optional
from transformers import CLIPTokenizer, CLIPTextModel

from time import sleep as delay

# Regex patterns for attention parsing
RE_ATTENTION = re.compile(
    r"\\[()\[\]]|\\\\|\\|[()\[\]]|:([+-]?[\d.]+)\)|[^\\()\[\]:]+|:",
    re.VERBOSE
)
RE_BREAK = re.compile(r"\s*\bBREAK\b\s*", re.S)

# BOS/EOS token IDs for Flux tokenizer
BOS_TOKEN_ID = 49406
EOS_TOKEN_ID = 49407
MAX_CHUNK = 75

def ensure_dtype(tensor, dtype):
    if tensor is None:
        return None
    return tensor if tensor.dtype == dtype else tensor.to(dtype=dtype)

def unpack_laten(images_latens, height: int, width: int, scale_factor:int) -> torch.Tensor:
    batch_size, num_patches, channels = images_latens.shape

    height = 2 * (int(height) // (scale_factor * 2))
    width  = 2 * (int(width) // (scale_factor * 2))

    view_latents = images_latens.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    permute_latents = view_latents.permute(0, 3, 1, 4, 2, 5)
    reshape_latents = permute_latents.reshape(batch_size, channels // (2 * 2), height, width)

    return reshape_latents

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
    
def parse_prompt_attention(text: str) -> List[Tuple[str, float]]:
    """
    Split `text` into (substring, weight) pairs based on bracketed attention syntax.
    """
    tokens: List[Tuple[str, float]] = []
    round_starts, square_starts = [], []
    mul_round, mul_square = 1.1, 1 / 1.1

    def apply_multiplier(start_idx: int, factor: float):
        for i in range(start_idx, len(tokens)):
            tokens[i] = (tokens[i][0], tokens[i][1] * factor)

    for match in RE_ATTENTION.finditer(text):
        segment, weight_tag = match.group(0), match.group(1)
        if segment.startswith('\\'):
            tokens.append((segment[1:], 1.0))
        elif segment == '(':
            round_starts.append(len(tokens))
        elif segment == '[':
            square_starts.append(len(tokens))
        elif weight_tag and round_starts:
            apply_multiplier(round_starts.pop(), float(weight_tag))
        elif segment == ')' and round_starts:
            apply_multiplier(round_starts.pop(), mul_round)
        elif segment == ']' and square_starts:
            apply_multiplier(square_starts.pop(), mul_square)
        else:
            parts = RE_BREAK.split(segment)
            for i, part in enumerate(parts):
                if i:
                    tokens.append(("BREAK", -1.0))
                if part:
                    tokens.append((part, 1.0))

    for idx in round_starts:
        apply_multiplier(idx, mul_round)
    for idx in square_starts:
        apply_multiplier(idx, mul_square)

    if not tokens:
        return [("", 1.0)]

    merged = [tokens[0]]
    for text_seg, w in tokens[1:]:
        prev_text, prev_w = merged[-1]
        if w == prev_w:
            merged[-1] = (prev_text + text_seg, prev_w)
        else:
            merged.append((text_seg, w))
    return merged


def tokenize_weights(tokenizer: CLIPTokenizer, segments: List[Tuple[str, float]], add_special_tokens: bool = False) -> Tuple[List[int], List[float]]:
    token_ids, weights = [], []
    for seg, weight in segments:
        enc = tokenizer(seg, truncation=True, max_length=77, return_overflowing_tokens=True, add_special_tokens=add_special_tokens)
        ids_list = enc.input_ids if isinstance(enc.input_ids[0], list) else [enc.input_ids]
        for ids in ids_list:
            if not add_special_tokens: ids = ids[1:-1]
            token_ids.extend(ids); weights.extend([weight] * len(ids))
    return token_ids, weights

def chunk_and_group(
    ids: List[int],
    weights: List[float],
    pad_last: bool = True
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Split tokens into chunks of MAX_CHUNK, prepend BOS/EOS, and pad if needed.
    """
    queue_ids = deque(ids)
    queue_w = deque(weights)
    chunks_ids, chunks_w = [], []

    while len(queue_ids) >= MAX_CHUNK:
        head_ids = [queue_ids.popleft() for _ in range(MAX_CHUNK)]
        head_w = [queue_w.popleft() for _ in range(MAX_CHUNK)]
        chunks_ids.append([BOS_TOKEN_ID] + head_ids + [EOS_TOKEN_ID])
        chunks_w.append([1.0] + head_w + [1.0])

    if queue_ids:
        remaining = list(queue_ids)
        remaining_w = list(queue_w)
        pad_len = MAX_CHUNK - len(remaining) if pad_last else 0
        chunk_ids = [BOS_TOKEN_ID] + remaining + [EOS_TOKEN_ID] * (pad_len + 1)
        chunk_w = [1.0] + remaining_w + [1.0] * (pad_len + 1)
        chunks_ids.append(chunk_ids)
        chunks_w.append(chunk_w)

    return chunks_ids, chunks_w


def get_flux1_embeddings(
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    prompt: str,
    negative_prompt: Optional[str] = None,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate CLIP and T5 embeddings and optionally clear encoder modules to free VRAM.
    """
    segments = parse_prompt_attention(prompt)
    ids, w = tokenize_weights(tokenizer, segments)
    groups, _ = chunk_and_group(ids.copy(), w.copy())
    clip_outputs = []
    for grp in groups:
        tensor = torch.tensor([grp], dtype=torch.long, device=device)
        with torch.no_grad():
            out = text_encoder(tensor)
        clip_outputs.append(out.pooler_output.squeeze(0))
    clip_embeds = torch.stack(clip_outputs, dim=0).mean(0, keepdim=True)

    neg = negative_prompt or prompt
    segments2 = parse_prompt_attention(neg)
    ids2, w2 = tokenize_weights(tokenizer_2, segments2, add_special_tokens=True)
    tensor2 = torch.tensor([ids2], dtype=torch.long, device=device)
    with torch.no_grad():
        out2 = text_encoder_2(tensor2)
    t5_embeds = out2.last_hidden_state
    for idx, weight in enumerate(w2):
        if weight != 1.0:
            t5_embeds[0, idx] *= weight

    return t5_embeds, clip_embeds
