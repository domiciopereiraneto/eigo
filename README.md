# EIGO

Evolutionary Image Generation Optimization (EIGO) is a backend and experiment suite for optimizing text-to-image generation through prompt embedding or latent noise search. The core engine combines diffusion pipelines, aesthetic predictors, CLIP-based prompt alignment, ImageReward, and HPSv2 to run black-box or gradient-based optimization without retraining the image model.

The repository currently exposes:

- `eigo.py`: the main backend engine (`Eigo`) used by all experiment scripts
- `src/optimization_targets.py`: shared prompt-embedding and latent-noise optimization target helpers
- `algorithms/eigo_single_prompt.py`: run one prompt with Adam, GA, CMA-ES, SNES, or CoSyNE
- `algorithms/p2_experiments.py`: run batches over sampled Parti Prompts or DrawBench categories
- `algorithms/p2_schedule.py`: launch multiple `p2_experiments.py` runs from scheduled config overrides
- `algorithms/optuna_schedule.py`: launch multiple Optuna searches from scheduled config overrides
- `algorithms/eigo_grid_search.py`: run parameter sweeps for one prompt
- `algorithms/tools/process_results.py`: generate summary tables, plots, and image grids from completed runs

## Setup

### Prerequisites

Install [Conda](https://docs.conda.io/) and clone the repository:

```bash
git clone <repository-url>
cd eigo
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate eigo
```

## Repository Layout

```text
eigo.py
src/
  optimization_targets.py
algorithms/
  config/
    config_eigo.yaml
    config_eigo_grid_search.yaml
    config_p2_experiments.yaml
    p2_schedule.yaml
  eigo_single_prompt.py
  eigo_grid_search.py
  p2_experiments.py
  p2_schedule.py
  tools/
    process_results.py
    config.yml
```

## Backend

`eigo.py` contains the `Eigo` class, which is the shared backend used by all experiment scripts. It is responsible for:

- loading the selected diffusion pipeline
- encoding prompt embeddings
- preparing the selected optimization target (`prompt_embeddings` or `latent_noise`)
- running Adam, GA, CMA-ES, SNES, or CoSyNE optimization
- scoring generated images with CLIP, an aesthetic predictor, ImageReward, and HPSv2
- saving images, metrics, and run configuration

The backend supports these diffusion backends:

- `sdxl` via `StableDiffusionXLPipeline`
- `sd` via `StableDiffusionPipeline` for Stable Diffusion v1/v2-family checkpoints
- `flux` via `FluxPipeline`
- `pixart` via `PixArtAlphaPipeline`
- `lcm` via Diffusers' `LatentConsistencyModelPipeline`/`DiffusionPipeline`
- `sana` via `SanaPipeline`
- `sana_sprint` via `SanaSprintPipeline`
- `auto` to infer the backend from `model_id`

Backends are configured through YAML, mainly with:

- `model_id`
- `model_backend`
- `optimization_target`
- `torch_dtype`
- `max_sequence_length`
- `use_multi_gpu`
- `pipeline_device_map`
- `max_memory`
- `enable_gradient_checkpointing`
- `enable_attention_slicing`
- `enable_vae_slicing`
- `enable_vae_tiling`

## Usage

### Notebook

For an interactive walkthrough:

```bash
jupyter notebook eigo_ex.ipynb
```

### Single-Prompt Optimization

Edit [`algorithms/config/config_eigo.yaml`](/home/posgrad/phd2025/dneto/eigo/algorithms/config/config_eigo.yaml) and run:

```bash
python algorithms/eigo_single_prompt.py
```

This script uses the backend in `eigo.py` and runs the method specified in `optimization_method`:

- `adam`
- `ga`
- `cmaes`
- `random_sampler`

Important fields in `config_eigo.yaml`:

- `selected_prompt`
- `optimization_method`
- `optimization_target`
- `seed`
- `cuda`
- `predictor`
- `num_inference_steps`
- `guidance_scale`
- `lcm_origin_steps` for LCM backends, typically `50`
- `aesthetic_score_weight`
- `clip_score_weight`
- `image_reward_score_weight`
- `hpsv2_score_weight`
- `pickscore_score_weight`
- `max_image_reward_score`
- `max_hpsv2_score`
- `max_pickscore_score`
- `image_reward_model`
- `hpsv2_version`
- `pickscore_model`
- `pickscore_processor`
- `evaluate_zero_weight_metrics`
- `results_folder`

By default, metrics with weight `0.0` are not loaded or evaluated. This keeps unused ImageReward, HPSv2, PickScore, and CLIP scorer models out of Adam VRAM. Set `evaluate_zero_weight_metrics: true` only when you need to log zero-weight metrics anyway.

Algorithm-specific fields:

- Adam: `num_iterations`, `adam_lr`, `adam_weight_decay`, `adam_eps`, `adam_beta1`, `adam_beta2`, `adam_max_grad_norm`
- GA: `num_generations`, `pop_size`, `ga_mutation_std`, `ga_elite_count`, `ga_crossover_operator`, `ga_crossover_rate`, `ga_crossover_alpha`, `ga_blend_alpha`, `ga_mutation_operator`, `ga_mutation_rate`, `save_gens`
- CMA-ES: `num_generations`, `pop_size`, `sigma`, `cmaes_variant`, `save_gens`
- SNES: `snes_num_generations`, `snes_pop_size`, `snes_sigma`, `snes_eta_mu`, `snes_eta_sigma`, `save_gens`
- CoSyNE: `cosyne_num_generations`, `cosyne_pop_size`, `cosyne_init_range`, `cosyne_mutation_probability`, `cosyne_mutation_scale`, `cosyne_parent_count`, `cosyne_offspring_count`, `save_gens`

Set `optimization_target: "prompt_embeddings"` to optimize text conditioning, or `optimization_target: "latent_noise"` to optimize/sample the initial diffusion latent noise vector. With `random_sampler`, `prompt_embeddings` fixes the initial latent noise and samples random prompt-embedding vectors, while `latent_noise` fixes the prompt embeddings and samples random latent-noise vectors.

For GA on continuous variables such as prompt embeddings or latent noise, use real-coded operators. Supported crossover operators are `uniform` for coordinate-wise parent mixing, `arithmetic` for convex interpolation, and `blend` for BLX-alpha style extrapolating interpolation. Supported mutation operators are `gaussian` for local perturbations, `cauchy` for occasional heavy-tailed jumps, and `none` for crossover-only ablations.

### Prompt Dataset Batch Experiments

Edit [`algorithms/config/config_p2_experiments.yaml`](/home/posgrad/phd2025/dneto/eigo/algorithms/config/config_p2_experiments.yaml) and run:

```bash
python algorithms/p2_experiments.py --config algorithms/config/config_p2_experiments.yaml
```

This script:

- loads the `nateraw/parti-prompts` dataset
- or loads the `sayakpaul/drawbench` dataset when `prompt_dataset: "drawbench"`
- samples prompts per category
- runs EIGO for each selected prompt
- saves run artifacts and aggregate plots/statistics inside the configured results directory

Relevant config keys:

- `prompt_per_categorie`
- `prompt_dataset`
- `prompt_sample_seed`
- `seed` or `seed_path`
- all shared backend/model parameters from `config_eigo.yaml`

### Scheduled Batch Runs

To launch multiple prompt dataset runs with different overrides:

```bash
python algorithms/p2_schedule.py \
  --base-config algorithms/config/config_p2_experiments.yaml \
  --schedule algorithms/config/p2_schedule.yaml
```

The schedule file can be either:

- a top-level list of override dictionaries
- a dictionary with a `runs` list

Each scheduled run is merged into the base config and executed sequentially.

### Grid Search

For one-prompt sweeps across Adam, GA, and/or CMA-ES parameter combinations, edit [`algorithms/config/config_eigo_grid_search.yaml`](/home/posgrad/phd2025/dneto/eigo/algorithms/config/config_eigo_grid_search.yaml) and run:

```bash
python algorithms/eigo_grid_search.py --config algorithms/config/config_eigo_grid_search.yaml
```

Useful options:

```bash
python algorithms/eigo_grid_search.py \
  --config algorithms/config/config_eigo_grid_search.yaml \
  --dry-run
```

The grid-search config supports:

- `selected_prompt`
- `test_adam`
- `test_ga`
- `test_cmaes`
- `test_random_sampler`
- shared backend/model parameters
- `grid.adam`, `grid.ga`, `grid.cmaes`, and `grid.random_sampler` parameter lists

Each run stores its effective parameter set and a summary YAML is written to the top-level results folder.

### Optuna Search

For adaptive hyperparameter search across Adam, GA, and/or CMA-ES, edit [`algorithms/config/config_eigo_optuna_search.yaml`](/home/posgrad/phd2025/dneto/eigo/algorithms/config/config_eigo_optuna_search.yaml) and run:

```bash
python algorithms/eigo_optuna_search.py --config algorithms/config/config_eigo_optuna_search.yaml
```

Useful options:

```bash
python algorithms/eigo_optuna_search.py \
  --config algorithms/config/config_eigo_optuna_search.yaml \
  --dry-run
```

The Optuna config supports the same base EIGO keys, including fixed `optimization_target`, plus `optuna.n_trials`, `optuna.direction`, `optuna.metric`, `optuna.sampler`, and method-specific `optuna.search_space` entries. Put `optimization_target` in a method search space to sample between `prompt_embeddings` and `latent_noise`. Lists are sampled as categorical choices, while dictionaries can define `float`, `int`, or `categorical` distributions. Each trial evaluates all selected prompts and optimizes the mean final objective value.

Search-space parameters can include `depends_on` to tune conditional parameters only when another sampled parameter has a matching `value` or one of several `values`. For example, `ga_crossover_alpha` can depend on `ga_crossover_operator: "arithmetic"`, while `ga_blend_alpha` can depend on `ga_crossover_operator: "blend"`.

To run several Optuna searches sequentially, recursively replacing base config values for each run:

```bash
python algorithms/optuna_schedule.py \
  --base-config algorithms/config/config_eigo_optuna_search.yaml \
  --schedule algorithms/config/optuna_schedule.yaml
```

Nested overrides are supported, including `optuna.study_name`, `optuna.storage`, and method-specific `optuna.search_space` values. Use distinct study names, storage URLs, and results folders for independent searches.

## Processing Results

After experiments finish, configure [`algorithms/tools/config.yml`](/home/posgrad/phd2025/dneto/eigo/algorithms/tools/config.yml) and run:

```bash
python algorithms/tools/process_results.py --config algorithms/tools/config.yml
```

This generates:

- summary tables
- evolution plots
- best-image grids
- prompt/category comparison views
- distance tables and grouped plots

## Output Structure

Runs are saved under the configured `results_folder`. The backend creates method- and model-specific experiment directories and stores per-run artifacts such as:

- `config.yaml`
- generated images
- score or fitness CSV files
- evolution plots
- best-image outputs

## License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
