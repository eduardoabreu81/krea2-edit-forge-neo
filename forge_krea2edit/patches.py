# Krea 2 Edit V2 for Forge Neo
# Two hooks, each handing everything to Forge Neo's own code while the extension is off for the generation: the Krea 2
# engine's conditioning (engine.py) and the Krea 2 DiT's forward (forward.py). Forge's files stay unchanged.

import functools

from backend.args import dynamic_args
from backend.diffusion_engine.krea import Krea2
from backend.nn import krea

from . import engine, state
from .forward import edit_forward

_applied = False


def _hook_forward() -> None:
    cls = krea.SingleStreamDiT
    original = cls.forward

    @functools.wraps(original)
    def forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        options = state.options
        if options is not None:
            size = (x.shape[-2], x.shape[-1])
            references, exact = state.lookup(size, dynamic_args.ref_latents)
            if references is not None:
                return edit_forward(self, x, timesteps, context, transformer_options, references, exact, options)
        return original(self, x, timesteps, context, attention_mask, transformer_options, **kwargs)

    cls.forward = forward


def apply() -> None:
    global _applied
    if _applied:
        return

    engine.install(Krea2)
    _hook_forward()

    _applied = True
    print("[Krea2 Edit V2] enabled: Krea 2 edits get the V2 reference controls (txt2img / img2img accordion)")
