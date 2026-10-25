"""Minimal stand-ins for the Forge Neo pieces the extension's testable modules use, for CPU tests without Forge.

Only the names the extension touches exist; anything else raises AttributeError, so a test notices when the
extension starts depending on more of Forge. Same shape as the FIBO 2 extension's forge_stubs.
"""

import math
import sys
import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def module(name, **attributes):
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    # backend.nn.flux.timestep_embedding (= ComfyUI's comfy.ldm.flux.layers.timestep_embedding)
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
                      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    # backend.utils.pad_to_patch_size (= ComfyUI's comfy.ldm.common_dit.pad_to_patch_size)
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad
    return F.pad(img, pad, mode=padding_mode)


def adaptive_resize(samples, width, height, upscale_method, crop):
    # backend.misc.image_resize.adaptive_resize, for 4D samples and the interpolate methods
    if crop == "center":
        old_width, old_height = samples.shape[-1], samples.shape[-2]
        old_aspect, new_aspect = old_width / old_height, width / height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        samples = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    return F.interpolate(samples, size=(height, width), mode=upscale_method)


def backend():
    """backend.nn.flux, backend.utils and backend.misc.image_resize, enough to run forge_krea2edit.forward."""
    flux = module("backend.nn.flux", timestep_embedding=timestep_embedding)
    nn = module("backend.nn", flux=flux)
    utils = module("backend.utils", pad_to_patch_size=pad_to_patch_size)
    image_resize = module("backend.misc.image_resize", adaptive_resize=adaptive_resize)
    misc = module("backend.misc", image_resize=image_resize)
    return {"backend": module("backend", nn=nn, utils=utils, misc=misc), "backend.nn": nn, "backend.nn.flux": flux,
            "backend.utils": utils, "backend.misc": misc, "backend.misc.image_resize": image_resize}


@contextmanager
def installed(stubs):
    """The stubs in sys.modules for the duration, and the extension's modules re-imported against them."""
    saved = {name: sys.modules.get(name) for name in stubs}
    ours = {name: mod for name, mod in sys.modules.items() if name.startswith("forge_krea2edit")}
    sys.modules.update(stubs)
    for name in ours:
        del sys.modules[name]
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        for name in [n for n in sys.modules if n.startswith("forge_krea2edit")]:
            del sys.modules[name]
        sys.modules.update(ours)
