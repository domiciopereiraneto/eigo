#!/usr/bin/env python3
"""Create prompt-by-optimization-step image grids for experiment folders."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import yaml
except ImportError:  # pragma: no cover - depends on local environment
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "config_optimization_step_grid.yaml"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True)
class Cell:
    path: Path
    label: str


@dataclass(frozen=True)
class PromptRow:
    prompt_index: int
    prompt_dir: Path
    prompt_text: str
    cells: list[Cell]


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read config files. Install project requirements "
            "or run `pip install PyYAML`."
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def as_int_list(config: dict, key: str, required: bool = False) -> list[int]:
    value = config.get(key)
    if value is None:
        if required:
            raise ValueError(f"Missing required config field: {key}")
        return []
    if not isinstance(value, list):
        raise ValueError(f"Config field {key} must be a list.")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field {key} must contain only integers.") from exc


def as_path_list(config: dict, key: str) -> list[Path]:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Missing required config field: {key}")
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config field {key} must be a non-empty list.")
    return [resolve_path(item) for item in value]


def numeric_suffix(text: str) -> Optional[int]:
    match = re.search(r"(\d+)$", text)
    return int(match.group(1)) if match else None


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: Optional[int] = None,
) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return [""]

    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if not line or text_size(draw, candidate, font)[0] <= max_width:
            line = candidate
            continue
        lines.append(line)
        line = word
        if max_lines is not None and len(lines) >= max_lines:
            break
    if max_lines is None or len(lines) < max_lines:
        lines.append(line)

    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
    if max_lines is not None and len(lines) == max_lines:
        while lines[-1] and text_size(draw, f"{lines[-1]}...", font)[0] > max_width:
            lines[-1] = lines[-1][:-1].rstrip()
        if lines[-1] and not text.endswith(lines[-1]):
            lines[-1] = f"{lines[-1]}..."
    return lines


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    font: ImageFont.ImageFont,
    x: int,
    y: int,
    width: int,
    fill: tuple[int, int, int],
    line_gap: int = 2,
) -> int:
    cursor = y
    for line in lines:
        line_w, line_h = text_size(draw, line, font)
        draw.text((x + (width - line_w) // 2, cursor), line, font=font, fill=fill)
        cursor += line_h + line_gap
    return cursor - y


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def image_candidates(path: Path) -> Iterable[Path]:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        yield path
        return
    for ext in IMAGE_EXTENSIONS:
        yield path.with_suffix(ext)


def first_existing_image(path_without_required_suffix: Path) -> Optional[Path]:
    for candidate in image_candidates(path_without_required_suffix):
        if candidate.exists():
            return candidate
    return None


def read_prompt_text(prompt_dir: Path) -> str:
    for csv_name in ("score_results.csv", "fitness_results.csv"):
        csv_path = prompt_dir / csv_name
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, nrows=1)
        except Exception:
            continue
        if "prompt" in df.columns and len(df) > 0:
            value = str(df.iloc[0]["prompt"]).strip()
            if value and value.lower() != "nan":
                return value
    return prompt_dir.name


def algorithm_key(experiment_dir: Path) -> str:
    name = experiment_dir.name.lower()
    for key in (
        "adam",
        "sepcmaes",
        "vdcmae",
        "cmaes",
        "cosyne",
        "snes",
        "zeroorder",
        "zero_order",
        "ga",
    ):
        if name.startswith(key) or f"_{key}_" in name:
            return key
    return name.split("_", 1)[0]


def find_run_dirs(path: Path) -> list[Path]:
    if any(path.glob("results_*")):
        return [path]
    run_dirs = [child for child in path.iterdir() if child.is_dir() and any(child.glob("results_*"))]
    return sorted(run_dirs, key=natural_key)


def find_prompt_dir(experiment_dir: Path, prompt_index: int) -> Optional[Path]:
    matches = []
    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        suffix = numeric_suffix(child.name)
        if suffix == prompt_index:
            matches.append(child)
    return sorted(matches, key=natural_key)[0] if matches else None


def available_population_ids(prompt_dir: Path, generation: int) -> list[int]:
    gen_dir = prompt_dir / f"gen_{generation}"
    if not gen_dir.exists():
        return []
    ids = []
    for image_path in gen_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = re.fullmatch(r"id_(\d+)", image_path.stem)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def infer_population_ids(prompt_dirs: Sequence[Path], steps: Sequence[int]) -> list[int]:
    for prompt_dir in prompt_dirs:
        for step in steps:
            ids = available_population_ids(prompt_dir, step)
            if ids:
                return ids
    return []


def build_cells(
    prompt_dir: Path,
    algo: str,
    steps: Sequence[int],
    population_ids: Sequence[int],
    include_baseline: bool,
    use_best_for_population_methods: bool,
) -> list[Cell]:
    cells: list[Cell] = []
    if include_baseline:
        baseline = first_existing_image(prompt_dir / "it_0")
        if baseline is None:
            baseline = first_existing_image(prompt_dir / "best_0")
        if baseline is not None:
            cells.append(Cell(baseline, "Baseline"))

    if algo == "adam":
        for step in steps:
            path = first_existing_image(prompt_dir / f"it_{step}")
            if path is not None:
                cells.append(Cell(path, f"Iter {step}"))
        return cells

    if use_best_for_population_methods:
        for step in steps:
            path = first_existing_image(prompt_dir / f"best_{step}")
            if path is not None:
                cells.append(Cell(path, f"Gen {step}"))
        return cells

    for step in steps:
        for pop_id in population_ids:
            path = first_existing_image(prompt_dir / f"gen_{step}" / f"id_{pop_id}")
            if path is not None:
                cells.append(Cell(path, f"G{step} P{pop_id}"))
    return cells


def create_grid(
    rows: Sequence[PromptRow],
    output_path: Path,
    tile_size: tuple[int, int],
    prompt_width: int,
    column_gap: int,
    row_gap: int,
    margin: int,
    header_height: int,
    prompt_font_size: int,
    header_font_size: int,
) -> None:
    if not rows:
        raise ValueError("No prompt rows to render.")

    max_cells = max(len(row.cells) for row in rows)
    if max_cells == 0:
        raise ValueError("No image cells to render.")

    tile_w, tile_h = tile_size
    grid_w = prompt_width + max_cells * tile_w + max(0, max_cells - 1) * column_gap
    grid_h = header_height + len(rows) * tile_h + max(0, len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (grid_w + 2 * margin, grid_h + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    prompt_font = load_font(prompt_font_size)
    header_font = load_font(header_font_size, bold=True)

    first_cells = rows[0].cells
    x0 = margin + prompt_width
    for col_idx, cell in enumerate(first_cells):
        label_lines = wrap_text(draw, cell.label, header_font, tile_w, max_lines=2)
        draw_centered_text(
            draw,
            label_lines,
            header_font,
            x0 + col_idx * (tile_w + column_gap),
            margin + 8,
            tile_w,
            (0, 0, 0),
        )

    y = margin + header_height
    for row in rows:
        prompt_lines = wrap_text(draw, row.prompt_text, prompt_font, prompt_width - 16, max_lines=8)
        prompt_text_h = sum(text_size(draw, line, prompt_font)[1] + 2 for line in prompt_lines)
        prompt_y = y + max(0, (tile_h - prompt_text_h) // 2)
        draw_centered_text(draw, prompt_lines, prompt_font, margin, prompt_y, prompt_width - 8, (0, 0, 0))

        for col_idx, cell in enumerate(row.cells):
            x = x0 + col_idx * (tile_w + column_gap)
            canvas.paste(fit_image(cell.path, tile_size), (x, y))
        y += tile_h + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"quality": 95} if output_path.suffix.lower() in {".jpg", ".jpeg"} else {}
    canvas.save(output_path, **save_kwargs)


def parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)x(\d+)", value.lower())
    if not match:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT, for example 160x160")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return width, height


def default_output_path(output_dir: Path, run_dir: Path) -> Path:
    return output_dir / f"{run_dir.name}_optimization_step_grid.jpg"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate image grids with prompts as rows and optimization steps as columns. "
            "For Adam, steps map to it_N images. For population methods, steps map to "
            "gen_N/id_M images unless --best is used."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG.relative_to(REPO_ROOT)}.",
    )
    args = parser.parse_args(argv)

    config_path = resolve_path(args.config)
    config = load_yaml(config_path)

    parent_dirs = as_path_list(config, "experiment_folders")
    missing = [path for path in parent_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment dirs: " + ", ".join(str(path) for path in missing))

    prompt_indices = as_int_list(config, "prompt_indices", required=True)
    steps = as_int_list(config, "steps", required=True)
    configured_population_ids = as_int_list(config, "population_ids")
    population_ids_configured = "population_ids" in config and config.get("population_ids") is not None

    run_dirs = []
    for path in parent_dirs:
        run_dirs.extend(find_run_dirs(path))
    run_dirs = list(dict.fromkeys(run_dirs))
    if not run_dirs:
        raise FileNotFoundError("No run dirs with results_* prompt folders were found.")

    output = config.get("output")
    if output and len(run_dirs) != 1:
        raise ValueError("--output can only be used when exactly one run dir is resolved.")

    output_dir = resolve_path(config.get("output_folder", "results_image_grids"))
    use_best = bool(config.get("use_best", False))
    include_baseline = bool(config.get("include_baseline", True))
    tile_size = parse_size(str(config.get("tile_size", "128x128")))
    prompt_width = int(config.get("prompt_width", 150))
    column_gap = int(config.get("column_gap", 3))
    row_gap = int(config.get("row_gap", 3))
    margin = int(config.get("margin", 10))
    header_height = int(config.get("header_height", 42))
    prompt_font_size = int(config.get("prompt_font_size", 14))
    header_font_size = int(config.get("header_font_size", 14))

    generated: list[Path] = []
    skipped: list[str] = []

    for run_dir in run_dirs:
        algo = algorithm_key(run_dir)
        prompt_dirs = []
        for prompt_index in prompt_indices:
            prompt_dir = find_prompt_dir(run_dir, prompt_index)
            if prompt_dir is None:
                skipped.append(f"{run_dir.name}: prompt {prompt_index} not found")
                continue
            prompt_dirs.append((prompt_index, prompt_dir))

        if not prompt_dirs:
            continue

        if use_best or algo == "adam":
            population_ids: list[int] = []
        elif population_ids_configured and configured_population_ids:
            population_ids = sorted(configured_population_ids)
        else:
            population_ids = infer_population_ids([prompt_dir for _, prompt_dir in prompt_dirs], steps)
            if not population_ids:
                skipped.append(f"{run_dir.name}: no population images found for selected steps")
                continue

        rows = []
        for prompt_index, prompt_dir in prompt_dirs:
            cells = build_cells(
                prompt_dir=prompt_dir,
                algo=algo,
                steps=steps,
                population_ids=population_ids,
                include_baseline=include_baseline,
                use_best_for_population_methods=use_best,
            )
            if not cells:
                skipped.append(f"{run_dir.name}: prompt {prompt_index} has no selected images")
                continue
            rows.append(PromptRow(prompt_index, prompt_dir, read_prompt_text(prompt_dir), cells))

        if not rows:
            continue

        output_path = resolve_path(output) if output else default_output_path(output_dir, run_dir)
        create_grid(
            rows=rows,
            output_path=output_path,
            tile_size=tile_size,
            prompt_width=prompt_width,
            column_gap=column_gap,
            row_gap=row_gap,
            margin=margin,
            header_height=header_height,
            prompt_font_size=prompt_font_size,
            header_font_size=header_font_size,
        )
        generated.append(output_path)
        print(f"Saved {output_path}")

    if skipped:
        print("Skipped:", file=sys.stderr)
        for item in skipped:
            print(f"  - {item}", file=sys.stderr)
    if not generated:
        raise RuntimeError("No grids were generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
