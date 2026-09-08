# EIGO

Evolutionary Image Generation Optimization (EIGO) is an experiment framework for inference-time optimization of text-to-image diffusion generation. It keeps the image generator fixed and searches over prompt embeddings or initial latent noise using gradient-based, evolutionary, and random-search baselines.

The core implementation is the `Eigo` class in [eigo.py](/home/posgrad/phd2025/dneto/eigo_old/eigo.py). The `algorithms/` directory contains runnable entrypoints for single-prompt runs, batched prompt-dataset experiments, hyperparameter search, scheduled runs, and result processing.

## Main Capabilities

- Optimize continuous prompt embeddings, latent noise, a joined noise/embedding vector, or separate noise/embedding sub-vectors for cooperative coevolution without retraining the diffusion model.
- Run AdamW, GA, CMA-ES, sep-CMA-ES, VD-CMA, CC-CMA-ES, CC-sep-CMA-ES, SNES, CC-SNES, CoSyNE, GOMEA, zero-order search, and random sampling.
- Score candidates with CLIPScore, LAION aesthetic predictors, ImageReward, HPSv2, PickScore, and JPEG-size objectives.
- Run prompt batches from Parti Prompts, DrawBench, or compatible Hugging Face datasets.
- Tune method hyperparameters with Optuna.
- Produce quantitative summaries, runcharts, optimization-step grids, and prompt-by-algorithm image grids.

## Repository Layout

```text
eigo.py                                      # Shared optimization backend
src/
  optimization_targets.py                    # Prompt-embedding, latent-noise, joint, and CC target helpers
  aesthetic_evaluation.py                    # Aesthetic predictor wrappers
algorithms/
  eigo_single_prompt.py                      # One prompt, one optimizer
  run_experiments.py                         # Prompt-dataset batch experiments
  experiments_schedule.py                    # Sequential batch-run launcher
  eigo_grid_search.py                        # Cartesian parameter sweeps
  eigo_optuna_search.py                      # Optuna hyperparameter search
  optuna_schedule.py                         # Sequential Optuna launcher
  process_quantitative_results.py            # Tables and spreadsheets
  process_runchart_results.py                # Metric-vs-NFE/time plots
  create_optimization_step_grid.py           # Prompt x optimization-step image grids
  create_prompt_algorithm_grid.py            # Prompt x algorithm best-image grids
  fill_zero_weight_metrics.py                # Backfill metrics that were skipped during optimization
  config/                                    # YAML configs for each entrypoint
environment.yml                              # Recommended Conda environment
requirements.txt                             # Pip dependency list
dependencies.yml                             # Full Conda export for stricter reproduction
```

## Setup

Create the recommended Conda environment:

```bash
conda env create -f environment.yml
conda activate eigo
```

If you prefer pip inside an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

The first run may download model checkpoints and prompt datasets from Hugging Face and the scorer packages. Make sure the target machine has enough disk space and GPU memory for the selected model backend.

## Configuration

Most scripts are configured with YAML files in [algorithms/config](/home/posgrad/phd2025/dneto/eigo_old/algorithms/config). The most commonly edited fields are:

- `model_id`, `model_backend`, `torch_dtype`, `height`, `width`, `num_inference_steps`, and `guidance_scale`
- `optimization_target`: `prompt_embeddings`, `latent_noise`, `noise_embeddings_flat`, or `noise_embeddings_cc`
- `optimization_method`: `adam`, `ga`, `cmaes`, `snes`, `cosyne`, `gomea`, `zero_order`, or `random_sampler`
- `aesthetic_predictor`: `0` for Simulacra, `1` for LAION V1, or `2` for LAION V2
- objective weights such as `clip_score_weight`, `image_reward_score_weight`, and `hpsv2_score_weight`
- method parameters such as `num_generations`, `pop_size`, `sigma`, `snes_sigma`, `cosyne_mutation_scale`, `gomea_init_range`, and `adam_lr`
- `results_folder`, which controls where artifacts are written

For CC-CMA-ES, set `optimization_method: cmaes`, `optimization_target: noise_embeddings_cc`, and `cmaes_variant` to `cc`, `cc_sep`, or `cc_vd`. For CC-SNES, set `optimization_method: snes` and `optimization_target: noise_embeddings_cc`. Both CC implementations keep the noise and embedding vectors as separate sub-populations and evaluate both sub-populations once per generation. Use `cc_embedding_sigma` and `cc_noise_sigma` to set separate initial search scales; either can be `null` or omitted to fall back to `sigma` for CMA-ES or `snes_sigma` for SNES. For CC-SNES, `cc_embedding_eta_mu`, `cc_embedding_eta_sigma`, `cc_noise_eta_mu`, and `cc_noise_eta_sigma` can set separate SNES adaptation rates; `null` or omitted values fall back to the shared `snes_eta_mu` and `snes_eta_sigma` settings.

Set `evaluate_zero_weight_metrics: false` to avoid loading scorer models whose weights are zero. This is useful for reducing AdamW memory use. Set it to `true` only when zero-weight metrics should still be logged for analysis.

## Single-Prompt Runs

Edit [algorithms/config/config_eigo.yaml](/home/posgrad/phd2025/dneto/eigo_old/algorithms/config/config_eigo.yaml), then run:

```bash
python algorithms/eigo_single_prompt.py
```

This runs one prompt with the optimizer selected in `optimization_method` and saves images, CSV metrics, plots, and the effective config under `results_folder`.

## Batch Experiments

Edit [algorithms/config/config_run_experiments.yaml](/home/posgrad/phd2025/dneto/eigo_old/algorithms/config/config_run_experiments.yaml), then run:

```bash
python algorithms/run_experiments.py --config algorithms/config/config_run_experiments.yaml
```

The batch runner samples prompts from `nateraw/parti-prompts` or `sayakpaul/drawbench`, or from another compatible Hugging Face dataset if `prompt_dataset_path` is provided. Use `use_entire_dataset`, `prompt_index_range`, `prompt_per_categorie`, `prompt_sample_seed`, `seed`, and `seed_path` to control prompt and seed selection.

To run several configured batches sequentially:

```bash
python algorithms/experiments_schedule.py \
  --base-config algorithms/config/config_run_experiments.yaml \
  --schedule algorithms/config/experiments_schedule.yaml
```

Each `runs` entry in the schedule is recursively merged onto the base config and launched as a separate batch.

## Hyperparameter Search

Edit [algorithms/config/config_eigo_optuna_search.yaml](/home/posgrad/phd2025/dneto/eigo_old/algorithms/config/config_eigo_optuna_search.yaml), then run:

```bash
python algorithms/eigo_optuna_search.py --config algorithms/config/config_eigo_optuna_search.yaml
```

Use `--dry-run` to print the enabled methods and search spaces without launching trials:

```bash
python algorithms/eigo_optuna_search.py \
  --config algorithms/config/config_eigo_optuna_search.yaml \
  --dry-run
```

The `optuna` section controls the study name, SQLite storage, sampler, prompt source, objective metric, trial count, and per-method search spaces. Lists are categorical choices; dictionaries can define `float`, `int`, or `categorical` distributions. Conditional parameters use `depends_on`.

To run several Optuna studies sequentially:

```bash
python algorithms/optuna_schedule.py \
  --base-config algorithms/config/config_eigo_optuna_search.yaml \
  --schedule algorithms/config/optuna_schedule.yaml
```

## Result Processing

Quantitative summaries:

```bash
python algorithms/process_quantitative_results.py \
  --config algorithms/config/config_process_quantitative_results.yaml
```

This writes `quantitative_summary_long.csv`, `quantitative_summary_wide.csv`, `quantitative_prompt_values.csv`, and an Excel workbook.

Runcharts:

```bash
python algorithms/process_runchart_results.py \
  --config algorithms/config/config_process_runchart_results.yaml
```

This writes metric curves by number of function evaluations and elapsed time.

Optimization-step image grids:

```bash
python algorithms/create_optimization_step_grid.py \
  --config algorithms/config/config_optimization_step_grid.yaml
```

Prompt-by-algorithm best-image grids:

```bash
python algorithms/create_prompt_algorithm_grid.py \
  --config algorithms/config/config_prompt_algorithm_grid.yaml
```

Backfill skipped zero-weight metrics for existing runs:

```bash
python algorithms/fill_zero_weight_metrics.py \
  --config algorithms/config/config_zero_weight_metrics.yaml
```

## Output Structure

Experiment folders contain one `results_*` directory per prompt. Depending on the optimizer, each prompt directory contains files such as:

- `config.yaml`
- `score_results.csv` for AdamW-style optimization
- `fitness_results.csv` for population and random-search methods
- generated images, including `it_0` baseline images and best/final outputs
- plots and summary images

Higher-level processing scripts read these prompt directories and write aggregate CSV, Excel, PNG, or JPG artifacts to the configured output folders.

## License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
