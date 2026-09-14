# EIGO: discrete text token optimization

EIGO optimizes the integer token IDs consumed by diffusion models' text encoders.
The supported methods are `ga`, `gomea`, and `random_sampler`; the only
`optimization_target` is `text_tokens`.

The existing SD (including DPO UNet replacements), SDXL, FLUX, PixArt, LCM,
Sana and Sana Sprint loaders remain available. Each pipeline performs its native
prompt preprocessing. EIGO captures the positive input IDs at each text encoder,
then replaces only non-special, non-padding positions with valid IDs from that
encoder's tokenizer vocabulary. Multi-encoder models use separate token segments
and vocabularies. Sequence lengths, attention masks, negative prompts and special
tokens remain fixed. For backends that add instruction text, non-special tokens
in that instruction are also part of the search space.

Candidates are encoded directly from IDs, without decoding and retokenizing.
The encoders and diffusion model are frozen. Every candidate in a run uses the
same diffusion seed; only the tokens change. Fitness compares the generated image
against the **original prompt**, using the configured aesthetic, CLIP, ImageReward,
HPSv2, PickScore and JPEG-size objectives. Candidate evaluation is sequential so
backend-specific masks and encoder state remain associated with each candidate.

## Setup and entry points

```bash
conda env create -f environment.yml
conda activate eigo
python algorithms/eigo_single_prompt.py
python algorithms/run_experiments.py --config algorithms/config/config_run_experiments.yaml
python algorithms/experiments_schedule.py
python algorithms/eigo_grid_search.py --config algorithms/config/config_eigo_grid_search.yaml
python algorithms/eigo_optuna_search.py --config algorithms/config/config_eigo_optuna_search.yaml
python algorithms/optuna_schedule.py
```

Model checkpoints and enabled scoring models must be available locally or
accessible through their providers. Set `cuda: cpu` for CPU execution. Single
prompt settings live in `algorithms/config/config_eigo.yaml`; batch settings live
in `config_run_experiments.yaml`. Schedules merge their overrides onto the
corresponding base config. Grid search and Optuna expose only the three methods.
Old continuous-search configs must be migrated; old optimization targets fail
validation.

## Search parameters

| Parameter | Meaning |
| --- | --- |
| `token_init_rate` | Probability of replacing each token when initializing GA/GOMEA populations (0–1). The original prompt is included. |
| `num_generations`, `pop_size` | GA generations and population size; `pop_size` also groups random samples for reporting. |
| `ga_mutation_rate` | Independent probability of vocabulary replacement per token. |
| `ga_mutation_operator` | `replacement` or `none`. |
| `ga_crossover_operator` | `uniform` or `one_point`. |
| `ga_crossover_rate` | Probability of inheriting from the first parent in uniform crossover. |
| `ga_elite_count` | Number of retained individuals; at least 1 and smaller than population size. Parents use binary tournament selection. |
| `gomea_num_generations`, `gomea_pop_size` | GOMEA generations and population size. |
| `gomea_linkage_model` | `linkage_tree`, `static_linkage_tree`, `univariate`, `full`, or `block_marginal_product`. |
| `gomea_bmp_block_size` | Positive block length for the block linkage model. |
| `gomea_max_evaluations` | Optional cap on candidate evaluations, excluding the baseline. Null uses the generation limit. |
| `num_images_to_generate` | Number of independent random token candidates, excluding the baseline. |
| `time_limit_seconds` | Optional search deadline checked between candidate evaluations. An in-flight generation call can finish after the deadline. |

GOMEA is implemented locally for categorical variables. Linkage trees use token
mutual information and average-linkage clustering. Optimal mixing accepts
non-worsening donor substitutions, with elitist forced improvement when mixing
stalls. A static tree is learned once; the ordinary tree is rebuilt each
generation. Tree construction can be costly for very long prompts; univariate
and block models avoid the pairwise linkage computation.

Random sampling draws each mutable position independently and uniformly from its
encoder's non-special vocabulary. It does not search random diffusion seeds.

## Results and replay

Each prompt run saves:

- `config.yaml`, `initial_tokens.json`, and `best_tokens.json` with complete
  per-encoder IDs, mutable positions and decoded text for inspection.
- `candidates.jsonl` with every evaluated candidate's integer vector, seed,
  generation and objective score.
- `it_0.jpg`, `best_N.jpg`, `best_all.jpg`, and optional `gen_N/id_M.jpg` images.
- `fitness_results.csv` and plots, including cumulative candidate `evaluations`
  and `best_fitness`. The baseline is excluded from that count; run charts add it
  when calculating diffusion function evaluations.

Decoded text is descriptive: re-tokenizing it may produce different IDs. Replay
an artifact using the same model, original prompt and configuration:

```python
import json
import numpy as np
from eigo import Eigo
from src.optimization_targets import TokenSpace

engine = Eigo(config)
artifact = json.load(open("best_tokens.json"))
space = TokenSpace(engine, artifact["original_prompt"])
pe, pooled = space.encode(np.asarray(artifact["tokens"], dtype=np.int64))
image = engine.generate_image_from_tensors(pe, pooled, config["seed"])
```

The processing, image-grid and zero-weight metric backfill scripts remain in
`algorithms/`; their matching configs point to token experiment folders. Report
readers can still read older CSV schemas, but no old optimizer can be launched.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use small local encoders and synthetic objectives; no checkpoint downloads
are needed. Full model runs additionally require the model weights and suitable
compute resources.
