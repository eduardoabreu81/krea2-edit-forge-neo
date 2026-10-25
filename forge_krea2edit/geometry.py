# Krea 2 Edit V2 for Forge Neo
# Where each reference picture goes: how it is fitted to the result's size before the VAE sees it, where its tokens
# sit in RoPE space next to the result, and how much the result attends to it. A port of comfyui-krea2edit 2.0.0
# (lbouaraba, Apache-2.0, see NOTICE), whose geometry the V2 LoRAs were trained on. No Forge imports: CPU-testable.

import math

import torch

ALIGNED = "aligned"
BESIDE = "beside"
BAG = "bag of features"
POSITIONS = (ALIGNED, BESIDE, BAG)

FIT = "fit"
CROP = "crop (legacy)"
FIT_MODES = (FIT, CROP)

# a near-matched aspect ratio fills the result's grid by a minimal center-crop instead of leaving a margin: target
# columns with nothing in the reference beside them get filled by repeating its edge (upstream, 2026-07-14)
CROP_TOLERANCE = 0.08
# the VAE's downscale and the DiT's patch: a latent is 1/8 of the picture, a token 2x2 latents
VAE_SCALE = 8
GRID = 16


def image_ids(bs: int, frame: int, h: int, w: int, device, offset_h: float = 0.0, offset_w: float = 0.0):
    """RoPE positions of an h x w token grid: (frame, row, column), rows and columns from the given offsets."""
    ids = torch.zeros(h, w, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(h, device=device, dtype=torch.float32) + offset_h)[:, None]
    ids[..., 2] = (torch.arange(w, device=device, dtype=torch.float32) + offset_w)[None, :]
    return ids.reshape(1, h * w, 3).repeat(bs, 1, 1)


def reference_ids(bs: int, grids: list[tuple[int, int]], h: int, w: int, centered: bool, device):
    """The positions of each reference grid, frames 1..N. centered (the fit mode): a reference smaller than the
    result's h x w grid keeps stride 1 and sits at its fractional center, since fit already resampled its pixels to
    the result's density; otherwise every reference starts at the top-left corner, as the crop mode trained."""
    ids = []
    for i, (gh, gw) in enumerate(grids):
        if centered:
            ids.append(image_ids(bs, i + 1, gh, gw, device, max(0.0, (h - gh) / 2), max(0.0, (w - gw) / 2)))
        else:
            ids.append(image_ids(bs, i + 1, gh, gw, device))
    return ids


def placements(count: int, subject: str, scene: str) -> list[str]:
    """One placement per reference: the last one is the subject (the only picture when there is one), the ones
    before it the scene, as upstream's ref_position and ref_position_a."""
    return [scene] * (count - 1) + [subject]


def place(ids: list[torch.Tensor], modes: list[str], h: int, w: int) -> list[torch.Tensor]:
    """Moves each reference on the row/column axes (frames untouched).
    aligned: reference token (r, c) on result token (r, c), how the LoRAs were trained;
    beside: to the right of the result (and of the references already placed beside it), its layout unchanged, so
            identity still flows but the layout pin is gone;
    bag of features (experimental): every token at the result's center, content without layout."""
    out, x0 = [], float(w)
    for positions, mode in zip(ids, modes):
        if mode == ALIGNED:
            out.append(positions)
            continue
        positions = positions.clone()
        if mode == BESIDE:
            positions[..., 2] = positions[..., 2] - positions[..., 2].min() + x0
            x0 = float(positions[..., 2].max()) + 1.0
        elif mode == BAG:
            positions[..., 1] = (h - 1) / 2.0
            positions[..., 2] = (w - 1) / 2.0
        else:
            raise ValueError(f"unknown reference placement {mode!r}")
        out.append(positions)
    return out


def fit_plan(ih: int, iw: int, lat_h: int, lat_w: int, mode: str) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """How an ih x iw picture becomes a reference for a result of lat_h x lat_w latents: the crop box (y0, x0, height,
    width) of the picture, in pixels, and the size it is resized to before the VAE.
    crop (legacy): center-crop to the result's proportions, resize to its exact size (v1 / v1.1 geometry).
    fit: a near-matched aspect ratio as crop; otherwise fit inside the result on the /16 grid, the axis that fits
    cropped so it lands on the grid without squashing (upstream's 2026-07-28 seam fix)."""
    px_h, px_w = lat_h * VAE_SCALE, lat_w * VAE_SCALE
    if mode == FIT:
        sc = min(px_h / ih, px_w / iw)
        if not (ih * sc >= px_h * (1 - CROP_TOLERANCE) and iw * sc >= px_w * (1 - CROP_TOLERANCE)):
            nh = min(max(GRID, int(ih * sc) // GRID * GRID), max(GRID, px_h // GRID * GRID))
            nw = min(max(GRID, int(iw * sc) // GRID * GRID), max(GRID, px_w // GRID * GRID))
            ch, cw = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            return ((ih - ch) // 2, (iw - cw) // 2, ch, cw), (nh, nw)
    elif mode != CROP:
        raise ValueError(f"unknown fit mode {mode!r}")
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    return ((ih - ch) // 2, (iw - cw) // 2, ch, cw), (px_h, px_w)


def fit_pixels(image: torch.Tensor, lat_h: int, lat_w: int, mode: str) -> torch.Tensor:
    """(B, H, W, C) in [0, 1] -> (B, h, w, 3) fitted as fit_plan says, bicubic with antialias as upstream."""
    (y0, x0, ch, cw), size = fit_plan(image.shape[1], image.shape[2], lat_h, lat_w, mode)
    pixels = image.movedim(-1, 1)[:, :3, y0:y0 + ch, x0:x0 + cw].float()
    pixels = torch.nn.functional.interpolate(pixels, size=size, mode="bicubic", antialias=True)
    return pixels.clamp(0, 1).movedim(1, -1)


def grounding_size(h: int, w: int, longest: int) -> tuple[int, int]:
    """The size the text encoder sees a picture at: its longest side capped to longest pixels (0 = as it is). The
    LoRAs trained with 384 to 768 pixels."""
    if longest and max(h, w) > longest:
        s = longest / max(h, w)
        return round(h * s), round(w * s)
    return h, w


def attention_bias(boosts: list[float], txtlen: int, ref_lens: list[int], tgtlen: int, device, dtype):
    """Additive attention-logit bias on the [text | references... | result] sequence: the result's queries attend to
    reference i's keys boosts[i] times as much, renormalized (log(boost) on the logits). None when no boost is on."""
    if all(b == 1.0 for b in boosts):
        return None
    rows = txtlen + sum(ref_lens)
    length = rows + tgtlen
    bias = torch.zeros(1, 1, length, length, device=device, dtype=dtype)
    start = txtlen
    for boost, ref_len in zip(boosts, ref_lens):
        if boost != 1.0:
            bias[:, :, rows:, start:start + ref_len] = math.log(max(boost, 1e-4))
        start += ref_len
    return bias
