# Flux VRAM-Optimized Image Generation

A modular, memory-efficient image generation pipeline built on top of Hugging Face Flux models. Designed to run on consumer-grade GPUs (12–16 GB VRAM) by splitting the workflow into three separate stages: prior embedding, latent generation, and VAE decoding. Utilizes `accelerate` with `device_map` to distribute workload across multiple GPUs.

---

## 🔗 Repository

[https://github.com/madtunebk/flux-vram-optimized](https://github.com/madtunebk/flux-vram-optimized)

---

## 🚀 Features

* **Modular Design**: Separate pipelines for embedding (FluxPriorRedux), latent generation (ImageGenerator), and decoding (ImageDecoder).
* **VRAM Optimization**: Only relevant components are loaded per stage to minimize peak memory usage.
* **Multi-GPU Support**: Requires at least two GPUs; uses `accelerate` and `device_map` under the hood.
* **Model Flexibility**: Switch between `black-forest-labs/FLUX.1-dev`, `FLUX.1-schnell`, or custom Flux variants.
* **Redux Variation Mode**: Optional stylized variations of input images by enabling Flux Redux.
* **Reproducibility**: Seeded random generators for deterministic outputs.

---

---

## 📋 Prerequisites

* **Hardware**:
  * Minimum: 2 GPUs with 12 GB VRAM each (suggested testing), or 1 GPU with ≥20 GB VRAM
  * Recommended: ≥16 GB VRAM across ≥2 GPUs for smooth operation.

* **Software**:
  * Python 3.10+
  * PyTorch 2.x with CUDA support
  * `accelerate` (for distributed device mapping)
  * `diffusers`, `transformers`, `torch`, and other dependencies (see `requirements.txt`).

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/madtunebk/flux-vram-optimized.git
cd flux-vram-optimized

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🗂️ Repository Structure

```text
flux-vram-optimized/
├── modules/
│   ├── FluxPriorRedux.py      # Encodes image into prompt embeddings
│   ├── ImageGenerator.py      # Latent generator (text2img / img2img)
│   └── ImageDecoder.py        # VAE decoder pipeline
│   └── helpers.py             # max_memory, fix_size, debug utilities
├── run_inference.py           # Main entry-point script
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔧 Configuration

All script options can be adjusted in `run_inference.py`:

| Parameter       | Description                                                                          | Default                            |
| --------------- | ------------------------------------------------------------------------------------ | ---------------------------------- |
| `model_id`      | Hugging Face model ID for Flux generation (e.g., `black-forest-labs/FLUX.1-schnell`) | `black-forest-labs/FLUX.1-schnell` |
| `enable_redux`  | Toggle Flux Redux embedding stage                                                    | `True`                             |
| `scale`         | Fraction of GPU memory to allocate (reserve \~1 GB for OS/GUI)                       | `0.90`                             |
| `num_inference` | Number of diffusion steps / iterations                                               | `25`                               |
| `device_map`    | Configuration for `accelerate` to split modules across GPUs                          | `balanced`                         |

Edit these values directly or add CLI flags as needed.

---

## 🚀 Usage

Simply run the entire pipeline end-to-end with:

```bash
python run_inference.py
```

All configuration—such as `enable_redux`, memory `scale`, number of inference steps, and model IDs—is controlled by editing the top section of `run_inference.py`.

---

## 📸 Examples

![Output Example 1](examples/output1.png)
![Output Example 2](examples/output2.png)

---

## 🛠️ Advanced Tips

* **Memory Scaling**: Adjust `scale` between `0.90` and `0.95` depending on your VRAM.
* **GPU Selection**: Use `CUDA_VISIBLE_DEVICES` environment variable to control which GPUs are used.
* **Custom Models**: Replace `model_id` in the script with any compatible Flux or Diffusers model.
* **Batch Processing**: Extend `run_inference.py` to process multiple images in a loop or directory.

---

## ❓ Troubleshooting

* **OOM Errors**: Lower `scale`, reduce `num_inference` steps.
* **Missing GPU**: Ensure `accelerate` is configured (`accelerate config`) and GPUs are visible.
* **Slow Performance**: Try disabling full pipeline loading (`full_load=False`) or lower resolution.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

* Open issues for bugs or feature requests
* Submit pull requests with clear descriptions and tests
* Improve documentation or add examples

Please follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
