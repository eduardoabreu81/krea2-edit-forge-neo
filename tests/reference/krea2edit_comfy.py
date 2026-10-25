# Transcription of comfyui-krea2edit 2.0.0 (https://github.com/lbouaraba/comfyui-krea2edit, commit
# 13ef60f20828949becada815d3cbc371eb5ccaab, __init__.py), Copyright Lars Bouaraba, Apache License 2.0: the edit forward
# and the pixel-path fitting, as the tests' reference. ComfyUI's imports are replaced by the stand-ins in
# tests/forge_stubs.py (the same functions); the bodies are upstream's, prints dropped.

import math

import torch
import torch.nn.functional as F
from einops import rearrange

from tests.forge_stubs import pad_to_patch_size, timestep_embedding


def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    off_h, off_w = max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


REF_POSITIONS = ["aligned (trained)", "beside (right of target)", "bag of features"]


def _place_refs(ref_ids, modes, h_, w_):
    if isinstance(modes, str):
        modes = [modes] * len(ref_ids)
    out, x0 = [], float(w_)
    for ids, mode in zip(ref_ids, modes):
        if mode.startswith("aligned"):
            out.append(ids)
            continue
        ids = ids.clone()
        if mode.startswith("beside"):
            ids[..., 2] = ids[..., 2] - ids[..., 2].min() + x0
            x0 = float(ids[..., 2].max()) + 1.0
        elif mode.startswith("bag"):
            ids[..., 1] = (h_ - 1) / 2.0
            ids[..., 2] = (w_ - 1) / 2.0
        else:
            raise ValueError(f"unknown ref_position {mode!r}")
        out.append(ids)
    return out


def _to_4d(v):
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _fit_src(src, H, W):
    sh, sw = src.shape[-2:]
    if (sh, sw) == (H, W):
        return src
    s = max(H / sh, W / sw)
    ch, cw = min(sh, int(round(H / s))), min(sw, int(round(W / s)))
    y0, x0 = (sh - ch) // 2, (sw - cw) // 2
    src = src[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(src.float(), size=(H, W), mode="bilinear")


def _fit_encode_image(image, vae, H, W, cache, key, fit_mode="crop"):
    key = key + (fit_mode,)
    if key in cache:
        return cache[key]
    px_h, px_w = H * 8, W * 8
    img = image.movedim(-1, 1)
    ih, iw = img.shape[-2:]
    if fit_mode == "fit":
        sc = min(px_h / ih, px_w / iw)
        CROP_TOL = 0.08
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., y0:y0 + ch, x0:x0 + cw]
            nh, nw = px_h, px_w
        else:
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            ch2, cw2 = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
            img = img[..., y0:y0 + ch2, x0:x0 + cw2]
        img = F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True)
        lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
        cache[key] = lat
        return lat
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    img = img[..., y0:y0 + ch, x0:x0 + cw]
    img = F.interpolate(img.float(), size=(px_h, px_w), mode="bicubic", antialias=True)
    lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
    cache[key] = lat
    return lat


def _ref_attn_bias(boosts, boost_mask, txtlen, slens, tgtlen, mask_hw, device, dtype):
    nsrc = len(slens)
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, sl = offs[i], slens[i]
        if boost_mask is not None and i == nsrc - 1 and mask_hw is not None:
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(mask[None].float(), size=mask_hw[i], mode="area")[0, 0]
            cols = off + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + sl, device=device)
        bias[:, :, rows0:, cols] = math.log(max(b, 1e-4))
    return bias


def krea2_edit_forward(m, x, timesteps, context, src_latent, transformer_options,
                       ref_boost=1.0, ref_boost_a=1.0, ref_boost_mask=None,
                       ref_native=False, pos_mode="anchor", ref_position="aligned (trained)",
                       ref_position_a="aligned (trained)"):
    patch = m.patch

    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    x = pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    src_list = src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
    srcs = []
    for sl in src_list:
        src = _to_4d(sl).to(x.device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        if not ref_native and src.shape[-2:] != (H, W):
            src = _fit_src(src, H, W).to(x.dtype)
        srcs.append(pad_to_patch_size(src, (patch, patch), padding_mode="replicate"))
    src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

    context = m._unpack_context(context)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    src_imgs = [m.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                for s_ in srcs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    srclen = sum(si.shape[1] for si in src_imgs)
    combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)

    device = combined.device
    if pos_mode == "stride1" and ref_native:
        ref_ids = [_imgids_offset(bs, i + 1, gh, gw, h_, w_, device)
                   for i, (gh, gw) in enumerate(src_grids)]
    else:
        ref_ids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(src_grids)]
    ref_ids = _place_refs(ref_ids, [ref_position_a] * (len(ref_ids) - 1) + [ref_position], h_, w_)
    pos = torch.cat([
        torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]
        + ref_ids
        + [_imgids(bs, 0, h_, w_, device)],
        dim=1)
    freqs = m.pe_embedder(pos)

    attn_bias = None
    if ref_boost != 1.0 or ref_boost_a != 1.0:
        boosts = [ref_boost_a] * (len(src_imgs) - 1) + [ref_boost]
        attn_bias = _ref_attn_bias(boosts, ref_boost_mask, txtlen,
                                   [si.shape[1] for si in src_imgs], tgtlen,
                                   src_grids, combined.device, combined.dtype)

    for block in m.blocks:
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out
