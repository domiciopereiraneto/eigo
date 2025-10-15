
import os
import re
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
    Compose a grid using Pillow and place the corresponding prompt above each row.
    - First column is baseline it_0.png from the first method dir.
    - Remaining columns use best_all.png from each method dir.
    - Titles show Aes/CLIP. First row also shows method names.
    """
    base = source_dirs[0]
    subdirs = [n for n in os.listdir(base) if os.path.isdir(os.path.join(base, n))]
    subdirs = sorted(subdirs, key=prompt_num)

    cols = len(source_dirs) + 1
    if not subdirs:
        raise FileNotFoundError("No prompt folders found.")

    # Collect rows
    rows = []
    prompts = []
    for folder_name in subdirs:
        folder_paths = [os.path.join(source_dirs[i], folder_name) for i in range(len(source_dirs))]
        if not all(os.path.isdir(p) for p in folder_paths):
            continue

        it0_path = os.path.join(folder_paths[0], "it_0.png")
        best_paths = [os.path.join(p, "best_all.png") for p in folder_paths]

        if not os.path.isfile(it0_path) or not all(os.path.isfile(p) for p in best_paths):
            continue

        # Scores and prompt
        aes0, clip0, fit0, pr = get_scores(folder_paths[0], "score_results.csv", idx=0)
        if aes0 is None or clip0 is None or pr is None:
            aes0, clip0, fit0, pr = get_scores(folder_paths[0], "fitness_results.csv", idx=0)

        scores = [(aes0, clip0, fit0)]
        for p in folder_paths:
            aes, clip, fit, _ = get_scores(p, "fitness_results.csv")
            if aes is None or clip is None:
                aes, clip, fit, _ = get_scores(p, "score_results.csv")
            scores.append((aes, clip, fit))

        rows.append((it0_path, best_paths, scores))
        prompts.append(pr if isinstance(pr, str) and pr.strip() else folder_name)

    if not rows:
        raise FileNotFoundError("No rows with valid images and scores found.")

    # Load one image to get size
    sample_img = Image.open(rows[0][0])
    if resize_to is None:
        tile_w, tile_h = sample_img.size
    else:
        tile_w, tile_h = resize_to

    # Fonts
    prompt_font = load_font(prompt_fontsize)
    title_font = load_font(title_fontsize)

    # Compute grid width
    grid_w = cols * tile_w + (cols - 1) * col_gap
    total_w = grid_w + 2 * margin
    if prompt_max_width is None:
        prompt_max_width = grid_w  # full row width

    # Prebuild column x positions
    col_x = [margin + c * (tile_w + col_gap) for c in range(cols)]

    # Build each row canvas with dynamic heights
    row_images = []
    for r_idx, (it0, bests, scores) in enumerate(rows):
        # Open and resize
        imgs = [Image.open(it0)] + [Image.open(p) for p in bests]
        if resize_to is not None:
            imgs = [im.resize((tile_w, tile_h), Image.LANCZOS) for im in imgs]

        # Determine maxes for color coding
        aes_list = [s[0] for s in scores]
        clip_list = [s[1] for s in scores]
        fit_list = [s[2] for s in scores]
        max_aes = max(aes_list)
        max_clip = max(clip_list)
        max_fit = max(fit_list)

        # Prompt wrapping and height
        tmp_canvas = Image.new("RGB", (total_w, 100), header_bg)
        tmp_draw = ImageDraw.Draw(tmp_canvas)
        lines = wrap_text_to_width(tmp_draw, prompts[r_idx], prompt_font, prompt_max_width)
        prompt_h = line_height(prompt_font) * len(lines)

        # Title heights (one or two lines for first row)
        if r_idx == 0:
            # method name + score on next line
            title_lines_per_col = [2] * cols
        else:
            title_lines_per_col = [1] * cols
        title_h = line_height(title_font) * max(title_lines_per_col)

        # Row canvas height
        vpad_prompt_to_titles = 6
        vpad_titles_to_img = 6
        row_h = prompt_h + vpad_prompt_to_titles + title_h + vpad_titles_to_img + tile_h

        row_img = Image.new("RGB", (total_w, row_h), header_bg)
        draw = ImageDraw.Draw(row_img)

        # Prompt
        _ = paste_centered_text(draw, lines, prompt_font, margin, margin + grid_w, 0, fill=(0,0,0))

        # Titles + images
        y_titles = prompt_h + vpad_prompt_to_titles
        y_img = y_titles + title_h + vpad_titles_to_img
        for c in range(cols):
            # title text
            aes, clip, fit = scores[c]
            color = (0,0,0)
            if fit == max_fit:
                color = (102, 0, 153)  # purple
            elif aes == max_aes:
                color = (200, 0, 0)    # red
            elif clip == max_clip:
                color = (0, 0, 200)    # blue

            if r_idx == 0:
                t1 = method_names[c]
                t2 = f"Aes: {aes:.2f}  CLIP: {clip:.2f} Fit: {fit:.2f}"
                lines_title = [t1, t2]
            else:
                lines_title = [f"Aes: {aes:.2f}  CLIP: {clip:.2f} Fit: {fit:.2f}"]

            # draw centered within the tile column region
            x_left = col_x[c]
            x_right = x_left + tile_w
            paste_centered_text(draw, lines_title, title_font, x_left, x_right, y_titles, fill=color)

            # paste image
            row_img.paste(imgs[c], (x_left, y_img))

        row_images.append(row_img)

    # Stack rows with row_gap
    total_h = sum(im.height for im in row_images) + row_gap * (len(row_images)-1) + 2 * margin
    canvas = Image.new("RGB", (total_w, total_h), header_bg)

    y = margin
    for r_im in row_images:
        canvas.paste(r_im, (0, y))
        y += r_im.height + row_gap

    # Save
    out_path = os.path.join(save_path, "generated_image_comparison_grid.png")
    os.makedirs(save_path, exist_ok=True)
    canvas.save(out_path, format="PNG")
    print(f"Grid saved to {out_path}")
    return out_path
