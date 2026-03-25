# EIGO
Evolutionary Image Generation Optimization (EIGO) is an engine for experimentation of diffusion-based generative models optimization through evolutionary prompt embedding search. It refines text-to-image generation using aesthetic and semantic metrics, enabling controllable, efficient, and black-box optimization without model retraining.

## Setup

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/) installed on your system.

### Environment Setup

1. Clone the repository:

   If you haven't cloned the repository yet, run:
   
   ```bash
   git clone --recursive <repository-url>
   ```

   If you've already cloned the repository without submodules, run:
   
   ```bash
   git submodule update --init --recursive
   ```

3. Create the Conda environment using the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

4. Activate the environment:
    ```bash
    conda activate eigo
    ```

## Usage

### Interactive Notebook

For an interactive introduction and demonstration, see the provided Jupyter notebook:

```bash
jupyter notebook eigo_ex.ipynb
```

The notebook walks through setting up the environment, running optimization algorithms, and visualizing results. It is ideal for experimentation and understanding EIGO's workflow without writing code from scratch.

The repository contains two main optimization algorithms for text-to-image generation using SDXL:

1. Adam optimization 
2. CMA-ES-based optimization

Run 
```` bash
# Set the configurations in algorithms/config/config_eigo.yaml

python algorithms/config/config_eigo.yaml
````

### Running Optimization Experiments

Each algorithm can be run for a set of Parti Prompts (P2) prompts using its respective Python script with a configuration file:

```bash
# Run Adam optimization
python algorithms/adam_p2.py --config algorithms/config/config_adam_p2.yaml

# Run CMA-ES optimization
python algorithms/cmaes_p2.py --config algorithms/config/config_cmaes_p2.yaml
```

These experiments are set to execute a number of runs (one per seed) for a set of 
Parti Prompt prompts.

### Running Scheduled P2 Batches

To run `algorithms/p2_experiments.py` multiple times with different parameter overrides, use:

```bash
python algorithms/p2_schedule.py \
  --base-config algorithms/config/config_p2_experiments.yaml \
  --schedule algorithms/config/p2_schedule.yaml
```

`algorithms/config/p2_schedule.yaml` contains a list of run dictionaries (under `runs`), where each dictionary overrides fields from the base config for one run.

### Processing Results

After running the optimization experiments, you can process and analyze the results using:

```bash
python algorithms/tools/process_results.py --config algorithms/tools/config.yml
```

This will generate:
- Summary tables comparing different methods
- Image grids showing the best results
- Results analysis per prompt and category

### Configuration Parameters

Common parameters across all algorithms (set in config file):

- `seed`: Random seed for reproducibility
- `seed_path`: Path to file containing multiple seeds (optional)
- `cuda`: GPU device number
- `predictor`: Aesthetic predictor model (0=SAM, 1=LAIONV1, 2=LAIONV2)
- `num_inference_steps`: Number of denoising steps for image generation
- `height`: Output image height
- `width`: Output image width
- `results_folder`: Output directory for results
- `model_id`: Diffusion model identifier (e.g., `stabilityai/sdxl-turbo`, `black-forest-labs/FLUX.1-schnell`, `PixArt-alpha/PixArt-XL-2-1024-MS`, or a Z Image checkpoint)
- `model_backend`: `"auto"`, `"sdxl"`, `"flux"`, `"pixart"`, or `"zimage"` (recommended: `"auto"` unless explicit override is needed)
- `torch_dtype`: `"auto"`, `"float32"`, `"float16"`, or `"bfloat16"`
- `max_sequence_length`: Tokenizer max length for FLUX and Z Image (ignored by SDXL/PixArt)
- `cfg_normalization`: Z Image CFG normalization toggle
- `cfg_truncation`: Z Image CFG truncation factor
- `use_safetensors`: Whether to request safetensors weights
- `use_multi_gpu`: Enable model sharding across multiple GPUs
- `pipeline_device_map`: Sharding strategy (e.g. `"balanced"` or `"auto"`)
- `max_memory`: Optional memory budget map per device for sharded loading
- `enable_gradient_checkpointing`: Reduce activation memory for Adam (slower)
- `enable_attention_slicing`: Reduce attention memory (slower)
- `enable_vae_slicing`: Reduce VAE decode memory
- `enable_vae_tiling`: Reduce VAE decode memory for larger images

Algorithm-specific parameters:

**Adam:**
- `num_iterations`: Number of optimization iterations
- `adam_lr`: Learning rate
- `adam_weight_decay`: Weight decay parameter
- `adam_eps`: Epsilon parameter

**CMA-ES:**
- `num_generations`: Number of generations
- `pop_size`: Population size
- `sigma`: Initial step size
- `cmaes_variant`: CMA-ES variant ("cmaes", "sep", or "vd")


[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
