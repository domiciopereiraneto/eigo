
import os
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ---------- Helpers ----------

def prompt_num(name: str) -> int:
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else 10**9

def get_scores(folder_path, csv_name, idx=-1):
    csv_path = os.path.join(folder_path, csv_name)
    prompt = None
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        if 'prompt' in df.columns:
            prompt = df.iloc[0]['prompt']
        if 'aesthetic_score' in df.columns and 'clip_score' in df.columns:
            row = df.iloc[idx]
            aesthetic_score = float(row['aesthetic_score'])
            clip_score = float(row['clip_score'])
            fitness = float(row['combined_score'])
        elif 'max_aesthetic_score' in df.columns and 'max_clip_score' in df.columns:
            row = df.iloc[idx]
            aesthetic_score = float(row['max_aesthetic_score'])
            clip_score = float(row['max_clip_score'])
            fitness = float(row['max_fitness'])
        else:
            return None, None, prompt
        return aesthetic_score, clip_score, fitness, prompt
    return None, None, None, prompt

def parse_weights(name: str):
    m = re.search(r"_a(\d+)_b(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def load_font(size: int, fallback: bool = True) -> ImageFont.FreeTypeFont:
    # Try common fonts, then default bitmap font
    for name in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    if fallback:
        return ImageFont.load_default()
    raise RuntimeError("No usable font found. Install DejaVuSans.ttf or provide a path.")

def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    line = words[0]
    for w in words[1:]:
        test = line + " " + w
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
            lines.append(line)
            line = w
    lines.append(line)
    return lines

def line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent + 2  # small padding

def paste_centered_text(draw: ImageDraw.ImageDraw, lines: List[str], font, x_left: int, x_right: int, y_top: int, fill: Tuple[int,int,int]=(0,0,0)) -> int:
    """Draw multi-line text centered between x_left and x_right starting at y_top.
    Returns total height used.
    """
    lh = line_height(font)
    width = x_right - x_left
    for i, ln in enumerate(lines):
        w = draw.textlength(ln, font=font)
        x = x_left + (width - int(w)) // 2
        draw.text((x, y_top + i*lh), ln, font=font, fill=fill)
    return lh * len(lines)

# ---------- Main ----------

def create_image_grid(source_dirs: List[str],
                      method_names: List[str],
                      save_path: str,
                      prompt_fontsize: int = 16,
                      title_fontsize: int = 14,
                      resize_to: Tuple[int,int] = None,
                      col_gap: int = 12,
                      row_gap: int = 18,
                      margin: int = 18,
                      prompt_max_width: int = None,
                      header_bg: Tuple[int,int,int] = (255,255,255)):
    """
    Compose one grid per weight combination.
    - First column is baseline it_0.png from the first method folder in that weight group.
    - Remaining columns use best_all.png from each method folder in that weight group.
    - Titles show Aes/CLIP/Fitness. First row also shows method names.
    """
    if not source_dirs:
        raise FileNotFoundError("No source directories provided.")

    # Map method label per source_dir index (supports old and new config lengths)
    if len(method_names) == len(source_dirs) + 1:
        baseline_label = method_names[0]
        method_labels = method_names[1:]
    elif len(method_names) == len(source_dirs):
        baseline_label = "Initial Embedding"
        method_labels = method_names
    else:
        baseline_label = "Initial Embedding"
        method_labels = [Path(d).name for d in source_dirs]

    # Group source dirs by weight pair
    grouped = {}
    for i, d in enumerate(source_dirs):
        name = Path(d).name
        w = parse_weights(name)
        if w is None:
            continue
        grouped.setdefault(w, []).append((i, d))

    if not grouped:
        raise FileNotFoundError("No source directories with weight pattern '_aXX_bYY' were found.")

    os.makedirs(save_path, exist_ok=True)
    out_paths = []

    for (a_int, b_int) in sorted(grouped.keys(), key=lambda x: (-x[0], x[1])):
        items = grouped[(a_int, b_int)]
        dirs_this = [d for _, d in items]
        labels_this = [method_labels[idx] if idx < len(method_labels) else Path(d).name for idx, d in items]
        col_labels = [baseline_label] + labels_this

        base_dir = dirs_this[0]
        subdirs = [n for n in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, n))]
        subdirs = sorted(subdirs, key=prompt_num)
        if not subdirs:
            continue

        cols = len(dirs_this) + 1
        rows = []
        prompts = []

        for folder_name in subdirs:
            folder_paths = [os.path.join(d, folder_name) for d in dirs_this]
            if not all(os.path.isdir(p) for p in folder_paths):
                continue

            it0_path = os.path.join(folder_paths[0], "it_0.png")
            best_paths = [os.path.join(p, "best_all.png") for p in folder_paths]
            if not os.path.isfile(it0_path) or not all(os.path.isfile(p) for p in best_paths):
                continue

            aes0, clip0, fit0, pr = get_scores(folder_paths[0], "score_results.csv", idx=0)
            if aes0 is None or clip0 is None or fit0 is None:
                aes0, clip0, fit0, pr = get_scores(folder_paths[0], "fitness_results.csv", idx=0)

            if aes0 is None or clip0 is None or fit0 is None:
                continue

            scores = [(aes0, clip0, fit0)]
            all_ok = True
            for p in folder_paths:
                aes, clip, fit, _ = get_scores(p, "fitness_results.csv")
                if aes is None or clip is None or fit is None:
                    aes, clip, fit, _ = get_scores(p, "score_results.csv")
                if aes is None or clip is None or fit is None:
                    all_ok = False
                    break
                scores.append((aes, clip, fit))
            if not all_ok:
                continue

            rows.append((it0_path, best_paths, scores))
            prompts.append(pr if isinstance(pr, str) and pr.strip() else folder_name)

        if not rows:
            continue

        sample_img = Image.open(rows[0][0])
        if resize_to is None:
            tile_w, tile_h = sample_img.size
        else:
            tile_w, tile_h = resize_to

        prompt_font = load_font(prompt_fontsize)
        title_font = load_font(title_fontsize)

        grid_w = cols * tile_w + (cols - 1) * col_gap
        total_w = grid_w + 2 * margin
        local_prompt_max_width = prompt_max_width if prompt_max_width is not None else grid_w
        col_x = [margin + c * (tile_w + col_gap) for c in range(cols)]

        row_images = []
        for r_idx, (it0, bests, scores) in enumerate(rows):
            imgs = [Image.open(it0)] + [Image.open(p) for p in bests]
            if resize_to is not None:
                imgs = [im.resize((tile_w, tile_h), Image.LANCZOS) for im in imgs]

            aes_list = [s[0] for s in scores]
            clip_list = [s[1] for s in scores]
            fit_list = [s[2] for s in scores]
            max_aes = max(aes_list)
            max_clip = max(clip_list)
            max_fit = max(fit_list)

            tmp_canvas = Image.new("RGB", (total_w, 100), header_bg)
            tmp_draw = ImageDraw.Draw(tmp_canvas)
            lines = wrap_text_to_width(tmp_draw, prompts[r_idx], prompt_font, local_prompt_max_width)
            prompt_h = line_height(prompt_font) * len(lines)

            title_h = line_height(title_font) * (2 if r_idx == 0 else 1)
            vpad_prompt_to_titles = 6
            vpad_titles_to_img = 6
            row_h = prompt_h + vpad_prompt_to_titles + title_h + vpad_titles_to_img + tile_h

            row_img = Image.new("RGB", (total_w, row_h), header_bg)
            draw = ImageDraw.Draw(row_img)
            paste_centered_text(draw, lines, prompt_font, margin, margin + grid_w, 0, fill=(0, 0, 0))

            y_titles = prompt_h + vpad_prompt_to_titles
            y_img = y_titles + title_h + vpad_titles_to_img

            for c in range(cols):
                aes, clip, fit = scores[c]
                color = (0, 0, 0)
                if fit == max_fit:
                    color = (102, 0, 153)
                elif aes == max_aes:
                    color = (200, 0, 0)
                elif clip == max_clip:
                    color = (0, 0, 200)

                if r_idx == 0:
                    lines_title = [col_labels[c], f"Aes: {aes:.2f}  CLIP: {clip:.2f} Fit: {fit:.2f}"]
                else:
                    lines_title = [f"Aes: {aes:.2f}  CLIP: {clip:.2f} Fit: {fit:.2f}"]

                x_left = col_x[c]
                x_right = x_left + tile_w
                paste_centered_text(draw, lines_title, title_font, x_left, x_right, y_titles, fill=color)
                row_img.paste(imgs[c], (x_left, y_img))

            row_images.append(row_img)

        if not row_images:
            continue

        total_h = sum(im.height for im in row_images) + row_gap * (len(row_images) - 1) + 2 * margin
        canvas = Image.new("RGB", (total_w, total_h), header_bg)
        y = margin
        for r_im in row_images:
            canvas.paste(r_im, (0, y))
            y += r_im.height + row_gap

        out_path = os.path.join(save_path, f"generated_image_comparison_grid_a{a_int}_b{b_int}.png")
        canvas.save(out_path, format="PNG")
        print(f"Grid saved to {out_path}")
        out_paths.append(out_path)

    if not out_paths:
        raise FileNotFoundError("No valid rows found to build image grids.")
    return out_paths
