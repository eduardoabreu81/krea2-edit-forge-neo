# Krea 2 Edit V2 for Forge Neo
# Forge Neo's Krea 2 DiT forward with the V2 reference controls: Forge's own edit path (text, then the references'
# clean tokens at frames 1..N, then the result at frame 0, only the result's tokens out), plus upstream's placement,
# fitted grids and reference boost (comfyui-krea2edit 2.0.0, krea2_edit_forward). Forge's imports are inside
# functions, so the CPU tests run it on stand-ins.

import torch
from einops import rearrange

from . import geometry
from .state import Options, ReferenceSet


def _forge():
    from backend.misc.image_resize import adaptive_resize
    from backend.nn.flux import timestep_embedding
    from backend.utils import pad_to_patch_size

    return adaptive_resize, timestep_embedding, pad_to_patch_size


def edit_forward(m, x, timesteps, context, transformer_options, references: ReferenceSet, exact: bool,
                 options: Options):
    """m is Forge's SingleStreamDiT (LoRA-patched), x the noisy result latent (B, C, 1, H, W), context the 12-layer
    Qwen3-VL conditioning. exact: the references were prepared for this result size (else they are resized to it,
    as Forge does, which also drops the fit mode's own grid)."""
    adaptive_resize, timestep_embedding, pad_to_patch_size = _forge()

    x = x.squeeze(2)
    bs, _, H_orig, W_orig = x.shape
    patch = m.patch

    x = pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    centered = references.centered and exact
    refs, grids = [], []
    for ref in references.latents:
        ref = ref.to(x)
        if ref.shape[0] != bs:
            ref = ref[:1].expand(bs, *ref.shape[1:])
        if not centered and ref.shape[-2:] != (H, W):
            ref = adaptive_resize(ref, W, H, "area", "center")
        ref = pad_to_patch_size(ref, (patch, patch), padding_mode="replicate")
        refs.append(ref)
        grids.append((ref.shape[-2] // patch, ref.shape[-1] // patch))

    img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    refs = [m.first(rearrange(r, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)) for r in refs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    ref_lens = [r.shape[1] for r in refs]
    reflen = sum(ref_lens)
    combined = torch.cat((context, *refs, img), dim=1)

    device = combined.device
    refids = geometry.reference_ids(bs, grids, h_, w_, centered, device)
    modes = geometry.placements(len(refids), options.subject_position, options.scene_position)
    refids = geometry.place(refids, modes, h_, w_)
    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    pos = torch.cat((txtpos, *refids, geometry.image_ids(bs, 0, h_, w_, device)), dim=1)
    freqs = m.pe_embedder(pos)

    boosts = [options.scene_boost] * (len(refs) - 1) + [options.subject_boost]
    bias = geometry.attention_bias(boosts, txtlen, ref_lens, imglen, device, combined.dtype)

    for block in m.blocks:
        combined = block(combined, tvec, freqs, bias, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + reflen : txtlen + reflen + imglen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]

    return out.unsqueeze(2)
