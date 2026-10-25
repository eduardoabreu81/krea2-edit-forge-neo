# Transcription of Forge Neo's Krea 2 DiT forward (https://github.com/Haoming02/sd-webui-forge-classic, branch neo,
# commit 0c388f8, backend/nn/krea.py, SingleStreamDiT.forward), AGPL-3.0: the edit path the extension takes over, as
# the tests' reference. dynamic_args.ref_latents is an argument here; Forge's imports are the stand-ins in
# tests/forge_stubs.py.

import torch
from einops import rearrange

from tests.forge_stubs import adaptive_resize, pad_to_patch_size, timestep_embedding


def _imgids(bs: int, frame: int, h_: int, w_: int, device: torch.device) -> torch.Tensor:
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def forward(self, x, timesteps, context, ref_latents, transformer_options={}):
    x = x.squeeze(2)
    bs, c, H_orig, W_orig = x.shape
    patch = self.patch

    x = pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    if (_edit := bool(ref_latents)) is True:
        refs = []
        ref_grids = []
        for ref in ref_latents:
            if x.shape[0] == 2:  # batch_cond_uncond
                ref = torch.cat((ref, ref), dim=0)
            if ref.shape != x.shape:
                ref = adaptive_resize(ref, W, H, "area", "center")
            refs.append(pad_to_patch_size(ref.to(x), (patch, patch), padding_mode="replicate"))
            ref_grids.append((ref.shape[-2] // patch, ref.shape[-1] // patch))

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    img = self.first(img)

    if _edit:
        refs = [rearrange(r, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch) for r in refs]
        refs = [self.first(r) for r in refs]

    t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec = self.tproj(t)

    context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = self.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    if _edit:
        reflen = sum(ref.shape[1] for ref in refs)
        combined = torch.cat((context, *refs, img), dim=1)
    else:
        reflen = 0
        combined = torch.cat((context, img), dim=1)

    device = combined.device
    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)

    if _edit:
        refids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(ref_grids)]
        imgids = _imgids(bs, 0, h_, w_, device)
        pos = torch.cat((txtpos, *refids, imgids), dim=1)
    else:
        imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
        imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
        imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
        pos = torch.cat((txtpos, imgpos), dim=1)

    freqs = self.pe_embedder(pos)

    for block in self.blocks:
        combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

    final = self.last(combined, t)
    out = final[:, txtlen + reflen : txtlen + reflen + imglen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
    out = out[:, :, :H_orig, :W_orig]

    return out.unsqueeze(2)
