#!/usr/bin/env python3
"""Create prompt-by-algorithm image grids from completed EIGO runs.

Each configured algorithm folder contributes one column or row. The grid compares
the best/final image for selected prompt indices and can optionally annotate each
cell with metric values read from the prompt CSV files.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
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
    metric_lines: tuple["MetricCaption", ...] = ()


@dataclass(frozen=True)
class MetricCaption:
    label: str
    value_text: str
    numeric_value: Optional[float]
    highlighted: bool = False

    @property
    def text(self) -> str:
        return f"{self.label}: {self.value_text}"


METRIC_SPECS = {
    "objective": ("Obj", "combined_score", "max_fitness"),
    "fitness": ("Fit", "combined_score", "max_fitness"),
    "combined_score": ("Obj", "combined_score", "max_fitness"),
    "max_fitness": ("Fit", "combined_score", "max_fitness"),
    "aesthetic": ("Aes", "aesthetic_score", "max_aesthetic_score"),
    "aesthetic_score": ("Aes", "aesthetic_score", "max_aesthetic_score"),
    "max_aesthetic_score": ("Aes", "aesthetic_score", "max_aesthetic_score"),
    "clip": ("CLIP", "clip_score", "max_clip_score"),
    "clip_score": ("CLIP", "clip_score", "max_clip_score"),
    "max_clip_score": ("CLIP", "clip_score", "max_clip_score"),
    "image_reward": ("IR", "image_reward_score", "max_image_reward_score"),
    "imagereward": ("IR", "image_reward_score", "max_image_reward_score"),
    "image_reward_score": ("IR", "image_reward_score", "max_image_reward_score"),
    "max_image_reward_score": ("IR", "image_reward_score", "max_image_reward_score"),
    "hps": ("HPS", "hpsv2_score", "max_hpsv2_score"),
    "hpsv2": ("HPS", "hpsv2_score", "max_hpsv2_score"),
    "hpsv2_score": ("HPS", "hpsv2_score", "max_hpsv2_score"),
    "max_hpsv2_score": ("HPS", "hpsv2_score", "max_hpsv2_score"),
    "pickscore": ("Pick", "pickscore_score", "max_pickscore_score"),
    "pick_score": ("Pick", "pickscore_score", "max_pickscore_score"),
    "pickscore_score": ("Pick", "pickscore_score", "max_pickscore_score"),
    "max_pickscore_score": ("Pick", "pickscore_score", "max_pickscore_score"),
    "jpeg": ("JPEG", "jpeg_size_kb", "min_jpeg_size_kb"),
    "jpeg_size_kb": ("JPEG", "jpeg_size_kb", "min_jpeg_size_kb"),
    "min_jpeg_size_kb": ("JPEG", "jpeg_size_kb", "min_jpeg_size_kb"),
}


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


def as_string_list(config: dict, key: str) -> list[str]:
    value = config.get(key, [])
    if value is None or value is False:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"Config field {key} must be a string, list, null, or false.")
    return [str(item).strip() for item in value if str(item).strip()]


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


def read_metric_csv(prompt_dir: Path) -> tuple[Optional[pd.DataFrame], str]:
    score_csv = prompt_dir / "score_results.csv"
    fitness_csv = prompt_dir / "fitness_results.csv"
    if score_csv.exists():
        return pd.read_csv(score_csv), "score"
    if fitness_csv.exists():
        return pd.read_csv(fitness_csv), "fitness"
    return None, ""


def select_metric_row(df: pd.DataFrame, kind: str, selector: str) -> pd.Series:
    selector = selector.lower().strip()
    if df.empty:
        raise ValueError("Cannot select metric row from an empty CSV.")
    if selector == "first":
        return df.iloc[0]
    if selector == "last":
        return df.iloc[-1]
    if selector == "best":
        objective_col = "combined_score" if kind == "score" else "max_fitness"
        if objective_col in df.columns:
            objective = pd.to_numeric(df[objective_col], errors="coerce")
            if objective.notna().any():
                return df.loc[objective.idxmax()]
        return df.iloc[-1]
    raise ValueError("metric_row_selector must be one of: first, last, best")


def metric_column(metric_name: str, kind: str, columns: Sequence[str]) -> tuple[str, Optional[str]]:
    key = metric_name.lower().strip()
    if key in METRIC_SPECS:
        label, score_col, fitness_col = METRIC_SPECS[key]
        column = score_col if kind == "score" else fitness_col
        if column in columns:
            return label, column
        fallback = fitness_col if kind == "score" else score_col
        if fallback in columns:
            return label, fallback
        return label, None

    label = metric_name.replace("_score", "").replace("_", " ").title()
    if metric_name in columns:
        return label, metric_name
    return label, None


def format_metric_value(value, precision: int) -> tuple[str, Optional[float]]:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a", None
    numeric_value = float(numeric)
    return f"{numeric_value:.{precision}f}", numeric_value


def read_metric_lines(
    prompt_dir: Path,
    metrics: Sequence[str],
    row_selector: str,
    precision: int,
) -> tuple[MetricCaption, ...]:
    if not metrics:
        return ()
    try:
        df, kind = read_metric_csv(prompt_dir)
        if df is None:
            return tuple(MetricCaption(metric, "n/a", None) for metric in metrics)
        row = select_metric_row(df, kind, row_selector)
        lines = []
        for metric in metrics:
            label, column = metric_column(metric, kind, df.columns)
            value_text, numeric_value = (
                ("n/a", None) if column is None else format_metric_value(row.get(column), precision)
            )
            lines.append(MetricCaption(label, value_text, numeric_value))
        return tuple(lines)
    except Exception:
        return tuple(MetricCaption(metric, "n/a", None) for metric in metrics)


def highlight_prompt_metric_maxima(cells_by_run: list[list[GridCell]]) -> list[list[GridCell]]:
    if not cells_by_run:
        return cells_by_run

    highlighted = [list(row) for row in cells_by_run]
    prompt_count = max((len(row) for row in highlighted), default=0)
    for prompt_pos in range(prompt_count):
        metric_count = max(
            (len(row[prompt_pos].metric_lines) for row in highlighted if prompt_pos < len(row)),
            default=0,
        )
        for metric_pos in range(metric_count):
            values = [
                row[prompt_pos].metric_lines[metric_pos].numeric_value
                for row in highlighted
                if prompt_pos < len(row)
                and metric_pos < len(row[prompt_pos].metric_lines)
                and row[prompt_pos].metric_lines[metric_pos].numeric_value is not None
            ]
            if not values:
                continue
            max_value = max(values)
            for row_idx, row in enumerate(highlighted):
                if prompt_pos >= len(row) or metric_pos >= len(row[prompt_pos].metric_lines):
                    continue
                metric = row[prompt_pos].metric_lines[metric_pos]
                if metric.numeric_value != max_value:
                    continue
                metrics = list(row[prompt_pos].metric_lines)
                metrics[metric_pos] = replace(metric, highlighted=True)
                highlighted[row_idx][prompt_pos] = replace(row[prompt_pos], metric_lines=tuple(metrics))

    return highlighted


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


def draw_metric_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[MetricCaption],
    font,
    bold_font,
    x: int,
    y: int,
    width: int,
) -> None:
    if not lines:
        return
    cursor = y
    for line in lines:
        selected_font = bold_font if line.highlighted else font
        line_w, line_h = text_size(draw, line.text, selected_font)
        draw.text((x + (width - line_w) // 2, cursor), line.text, font=selected_font, fill=(45, 45, 45))
        cursor += line_h + 2


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
    layout: str,
    prompt_label_width: int,
    algorithm_header_height: int,
    image_metrics: Sequence[str],
    metric_row_selector: str,
    baseline_metric_row_selector: str,
    metric_font_size: int,
    metric_precision: int,
    metric_gap: int,
) -> tuple[int, int]:
    valid_layouts = {"prompt_x_algorithm", "algorithm_x_prompt"}
    if layout not in valid_layouts:
        raise ValueError(f"layout must be one of {sorted(valid_layouts)}, got: {layout}")

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
                    metric_lines=read_metric_lines(
                        prompt_dir,
                        image_metrics,
                        baseline_metric_row_selector,
                        metric_precision,
                    ),
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
                    metric_lines=read_metric_lines(
                        prompt_dir,
                        image_metrics,
                        metric_row_selector,
                        metric_precision,
                    ),
                )
            )
        row_labels.append(run.label)
        cells_by_run.append(row)

    if image_metrics:
        cells_by_run = highlight_prompt_metric_maxima(cells_by_run)

    tile_w, tile_h = tile_size
    prompt_font = load_font(prompt_font_size, bold=True)
    algorithm_font = load_font(algorithm_font_size, bold=True)
    metric_font = load_font(metric_font_size)
    metric_font_bold = load_font(metric_font_size, bold=True)
    tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    metric_line_h = max(
        text_size(tmp_draw, "0", metric_font)[1],
        text_size(tmp_draw, "0", metric_font_bold)[1],
    ) + 2
    metric_h = len(image_metrics) * metric_line_h if image_metrics else 0
    tile_block_h = tile_h + (metric_gap + metric_h if image_metrics else 0)

    if layout == "prompt_x_algorithm":
        width = algorithm_label_width + len(prompt_indices) * tile_w + max(0, len(prompt_indices) - 1) * column_gap
        row_count = len(cells_by_run)
        height = prompt_header_height + row_count * tile_block_h + max(0, row_count - 1) * row_gap
        canvas = Image.new("RGBA", (width + 2 * margin, height + 2 * margin), "white")
        draw = ImageDraw.Draw(canvas)

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
                    tile_block_h,
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
                    y + max(0, (tile_block_h - text_h) // 2),
                    algorithm_label_width - 8,
                    (0, 0, 0),
                )

            for col_idx, cell in enumerate(cells_by_run[row_idx]):
                x = x0 + col_idx * (tile_w + column_gap)
                if cell.image_path is None:
                    draw_missing_tile(draw, x, y, tile_size, algorithm_font)
                else:
                    canvas.paste(fit_image(cell.image_path, tile_size), (x, y))
                draw_metric_lines(
                    draw,
                    cell.metric_lines,
                    metric_font,
                    metric_font_bold,
                    x,
                    y + tile_h + metric_gap,
                    tile_w,
                )
            y += tile_block_h + row_gap
    else:
        width = prompt_label_width + len(row_labels) * tile_w + max(0, len(row_labels) - 1) * column_gap
        row_count = len(prompt_indices)
        height = algorithm_header_height + row_count * tile_block_h + max(0, row_count - 1) * row_gap
        canvas = Image.new("RGBA", (width + 2 * margin, height + 2 * margin), "white")
        draw = ImageDraw.Draw(canvas)

        x0 = margin + prompt_label_width
        for col_idx, label in enumerate(row_labels):
            x = x0 + col_idx * (tile_w + column_gap)
            lines = wrap_text(draw, label, algorithm_font, tile_w, max_lines=4)
            draw_centered_text(draw, lines, algorithm_font, x, margin + 4, tile_w, (0, 0, 0))

        y = margin + algorithm_header_height
        for prompt_idx, prompt_index in enumerate(prompt_indices):
            prompt_text = prompt_texts.get(prompt_index, str(prompt_index))
            lines = wrap_text(draw, prompt_text, prompt_font, prompt_label_width - 10, max_lines=5)
            text_h = sum(text_size(draw, line, prompt_font)[1] + 2 for line in lines)
            draw_centered_text(
                draw,
                lines,
                prompt_font,
                margin,
                y + max(0, (tile_block_h - text_h) // 2),
                prompt_label_width - 8,
                (0, 0, 0),
            )

            for col_idx, row in enumerate(cells_by_run):
                x = x0 + col_idx * (tile_w + column_gap)
                cell = row[prompt_idx]
                if cell.image_path is None:
                    draw_missing_tile(draw, x, y, tile_size, algorithm_font)
                else:
                    canvas.paste(fit_image(cell.image_path, tile_size), (x, y))
                draw_metric_lines(
                    draw,
                    cell.metric_lines,
                    metric_font,
                    metric_font_bold,
                    x,
                    y + tile_h + metric_gap,
                    tile_w,
                )
            y += tile_block_h + row_gap

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
        layout=str(config.get("layout", "prompt_x_algorithm")).lower(),
        prompt_label_width=int(config.get("prompt_label_width", 220)),
        algorithm_header_height=int(config.get("algorithm_header_height", config.get("prompt_header_height", 86))),
        image_metrics=as_string_list(config, "image_metrics"),
        metric_row_selector=str(config.get("metric_row_selector", "last")),
        baseline_metric_row_selector=str(config.get("baseline_metric_row_selector", "first")),
        metric_font_size=int(config.get("metric_font_size", 12)),
        metric_precision=int(config.get("metric_precision", 3)),
        metric_gap=int(config.get("metric_gap", 4)),
    )
    print(f"Saved {output_path} {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
