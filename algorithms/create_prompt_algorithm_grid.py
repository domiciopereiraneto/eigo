#!/usr/bin/env python3
"""Create prompt-by-algorithm grids from each run's best prompt image."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw

from create_optimization_step_grid import (
    REPO_ROOT,
    draw_centered_text,
    find_prompt_dir,
    find_run_dirs,
    first_existing_image,
    fit_image,
    load_font,
    load_yaml,
    parse_size,
    read_prompt_text,
    resolve_path,
    text_size,
    wrap_text,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "config_prompt_algorithm_grid.yaml"


@dataclass(frozen=True)
class AlgorithmRun:
    label: str
    run_dir: Path


@dataclass(frozen=True)
class GridCell:
    prompt_index: int
    prompt_text: str
    image_path: Optional[Path]


def numeric_stem_value(path: Path, prefix: str) -> Optional[int]:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def latest_prefixed_image(prompt_dir: Path, prefix: str) -> Optional[Path]:
    candidates = []
    for path in prompt_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        value = numeric_stem_value(path, prefix)
        if value is not None:
            candidates.append((value, path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def best_image(prompt_dir: Path, best_image_name: str, fallback_to_latest: bool) -> Optional[Path]:
    configured = first_existing_image(prompt_dir / best_image_name)
    if configured is not None:
        return configured
    if not fallback_to_latest:
        return None
    return latest_prefixed_image(prompt_dir, "best") or latest_prefixed_image(prompt_dir, "it")


def as_int_list(config: dict, key: str) -> list[int]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config field {key} must be a non-empty list.")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field {key} must contain only integers.") from exc


def algorithm_runs(config: dict) -> list[AlgorithmRun]:
    entries = config.get("algorithms")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Config field algorithms must be a non-empty list.")

    runs = []
    for idx, entry in enumerate(entries, start=1):
        if isinstance(entry, str):
            label = Path(entry).name
            folder = entry
        elif isinstance(entry, dict):
            folder = entry.get("folder")
            label = entry.get("label")
            if not folder:
                raise ValueError(f"Algorithm entry {idx} is missing folder.")
            if not label:
                label = Path(str(folder)).name
        else:
            raise ValueError(f"Algorithm entry {idx} must be a string or mapping.")

        resolved = resolve_path(folder)
        if not resolved.exists():
            raise FileNotFoundError(f"Algorithm folder does not exist: {resolved}")
        run_dirs = find_run_dirs(resolved)
        if len(run_dirs) != 1:
            raise ValueError(
                f"Algorithm folder must resolve to exactly one run dir, got {len(run_dirs)}: {resolved}"
            )
        runs.append(AlgorithmRun(str(label), run_dirs[0]))
    return runs


def draw_rotated_label(
    canvas: Image.Image,
    label: str,
    font,
    x: int,
    y: int,
    width: int,
    height: int,
    fill: tuple[int, int, int],
) -> None:
    label_img = Image.new("RGBA", (height, width), (255, 255, 255, 0))
    draw = ImageDraw.Draw(label_img)
    lines = wrap_text(draw, label, font, height - 8, max_lines=4)
    text_h = sum(text_size(draw, line, font)[1] + 2 for line in lines)
    draw_centered_text(draw, lines, font, 0, max(0, (width - text_h) // 2), height, fill)
    rotated = label_img.rotate(90, expand=True)
    canvas.alpha_composite(rotated, (x, y))


def draw_missing_tile(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    size: tuple[int, int],
    font,
) -> None:
    tile_w, tile_h = size
    draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), fill=(245, 245, 245), outline=(180, 180, 180))
    draw_centered_text(draw, ["Missing"], font, x, y + tile_h // 2 - 8, tile_w, (90, 90, 90))


def create_grid(
    runs: Sequence[AlgorithmRun],
    prompt_indices: Sequence[int],
    output_path: Path,
    best_image_name: str,
    fallback_to_latest: bool,
    include_baseline: bool,
    baseline_label: str,
    tile_size: tuple[int, int],
    algorithm_label_width: int,
    prompt_header_height: int,
    column_gap: int,
    row_gap: int,
    margin: int,
    prompt_font_size: int,
    algorithm_font_size: int,
    rotate_algorithm_labels: bool,
) -> tuple[int, int]:
    prompt_texts: dict[int, str] = {}
    row_labels: list[str] = []
    cells_by_run: list[list[GridCell]] = []
    skipped: list[str] = []

    if include_baseline:
        baseline_row = []
        baseline_source = runs[0]
        for prompt_index in prompt_indices:
            prompt_dir = find_prompt_dir(baseline_source.run_dir, prompt_index)
            if prompt_dir is None:
                skipped.append(f"{baseline_label}: prompt {prompt_index} not found")
                baseline_row.append(GridCell(prompt_index, prompt_texts.get(prompt_index, str(prompt_index)), None))
                continue
            prompt_text = prompt_texts.setdefault(prompt_index, read_prompt_text(prompt_dir))
            baseline_row.append(
                GridCell(
                    prompt_index=prompt_index,
                    prompt_text=prompt_text,
                    image_path=first_existing_image(prompt_dir / "it_0"),
                )
            )
        row_labels.append(baseline_label)
        cells_by_run.append(baseline_row)

    for run in runs:
        row = []
        for prompt_index in prompt_indices:
            prompt_dir = find_prompt_dir(run.run_dir, prompt_index)
            if prompt_dir is None:
                skipped.append(f"{run.label}: prompt {prompt_index} not found")
                row.append(GridCell(prompt_index, prompt_texts.get(prompt_index, str(prompt_index)), None))
                continue
            prompt_text = prompt_texts.setdefault(prompt_index, read_prompt_text(prompt_dir))
            row.append(
                GridCell(
                    prompt_index=prompt_index,
                    prompt_text=prompt_text,
                    image_path=best_image(prompt_dir, best_image_name, fallback_to_latest),
                )
            )
        row_labels.append(run.label)
        cells_by_run.append(row)

    tile_w, tile_h = tile_size
    width = algorithm_label_width + len(prompt_indices) * tile_w + max(0, len(prompt_indices) - 1) * column_gap
    row_count = len(cells_by_run)
    height = prompt_header_height + row_count * tile_h + max(0, row_count - 1) * row_gap
    canvas = Image.new("RGBA", (width + 2 * margin, height + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    prompt_font = load_font(prompt_font_size, bold=True)
    algorithm_font = load_font(algorithm_font_size, bold=True)

    x0 = margin + algorithm_label_width
    for col_idx, prompt_index in enumerate(prompt_indices):
        prompt_text = prompt_texts.get(prompt_index, str(prompt_index))
        x = x0 + col_idx * (tile_w + column_gap)
        lines = wrap_text(draw, prompt_text, prompt_font, tile_w, max_lines=4)
        draw_centered_text(draw, lines, prompt_font, x, margin + 4, tile_w, (0, 0, 0))

    y = margin + prompt_header_height
    for row_idx, label in enumerate(row_labels):
        if rotate_algorithm_labels:
            draw_rotated_label(
                canvas,
                label,
                algorithm_font,
                margin,
                y,
                algorithm_label_width,
                tile_h,
                (0, 0, 0),
            )
        else:
            lines = wrap_text(draw, label, algorithm_font, algorithm_label_width - 10, max_lines=4)
            text_h = sum(text_size(draw, line, algorithm_font)[1] + 2 for line in lines)
            draw_centered_text(
                draw,
                lines,
                algorithm_font,
                margin,
                y + max(0, (tile_h - text_h) // 2),
                algorithm_label_width - 8,
                (0, 0, 0),
            )

        for col_idx, cell in enumerate(cells_by_run[row_idx]):
            x = x0 + col_idx * (tile_w + column_gap)
            if cell.image_path is None:
                draw_missing_tile(draw, x, y, tile_size, algorithm_font)
            else:
                canvas.paste(fit_image(cell.image_path, tile_size), (x, y))
        y += tile_h + row_gap

    if skipped:
        print("Skipped:", file=sys.stderr)
        for item in skipped:
            print(f"  - {item}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = canvas.convert("RGB")
    save_kwargs = {"quality": 95} if output_path.suffix.lower() in {".jpg", ".jpeg"} else {}
    rgb.save(output_path, **save_kwargs)
    return rgb.size


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a prompt x algorithm grid from best prompt images.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG.relative_to(REPO_ROOT)}.",
    )
    args = parser.parse_args(argv)

    config = load_yaml(resolve_path(args.config))
    runs = algorithm_runs(config)
    prompt_indices = as_int_list(config, "prompt_indices")
    output_path = resolve_path(config.get("output", "results_image_grids/prompt_x_algorithm_best.jpg"))
    size = create_grid(
        runs=runs,
        prompt_indices=prompt_indices,
        output_path=output_path,
        best_image_name=str(config.get("best_image_name", "best_all")),
        fallback_to_latest=bool(config.get("fallback_to_latest", True)),
        include_baseline=bool(config.get("include_baseline", True)),
        baseline_label=str(config.get("baseline_label", "Baseline")),
        tile_size=parse_size(str(config.get("tile_size", "160x160"))),
        algorithm_label_width=int(config.get("algorithm_label_width", 90)),
        prompt_header_height=int(config.get("prompt_header_height", 86)),
        column_gap=int(config.get("column_gap", 10)),
        row_gap=int(config.get("row_gap", 3)),
        margin=int(config.get("margin", 12)),
        prompt_font_size=int(config.get("prompt_font_size", 14)),
        algorithm_font_size=int(config.get("algorithm_font_size", 16)),
        rotate_algorithm_labels=bool(config.get("rotate_algorithm_labels", True)),
    )
    print(f"Saved {output_path} {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
