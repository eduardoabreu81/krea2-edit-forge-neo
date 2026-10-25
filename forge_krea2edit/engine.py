# Krea 2 Edit V2 for Forge Neo
# The conditioning side of an edit, in place of Forge Neo's Krea2.get_learned_conditioning while the extension is on.
# Forge keeps the pictures to edit as pixels until the prompt is encoded (img2img's in ini_latent, ImageStitch's in
# ref_latents); here they become the V2 references: fitted in pixels to the result's size before the VAE (Forge
# encodes them at their own size and resizes the latent), shown to Qwen3-VL at the grounding size (Forge: 768x768 of
# area), and shown to it for the negative prompt too (Forge: text only).

import torch
from backend import memory_management
from backend.args import dynamic_args

from . import geometry, state
from .state import ReferenceSet


def _encode(engine, pixels: torch.Tensor) -> torch.Tensor:
    """(1, H, W, 3) in [0, 1] -> the latent the DiT reads (1, 16, H / 8, W / 8), as Forge's Krea2.encode_vision."""
    vae = engine.forge_objects.vae
    sample = vae.encode(pixels)
    return vae.first_stage_model.process_in(sample).squeeze(2)


def _grounding(picture: torch.Tensor, longest: int) -> torch.Tensor:
    """(1, H, W, C) -> (1, h, w, 3), the longest side capped as geometry.grounding_size says (area, as upstream)."""
    h, w = geometry.grounding_size(picture.shape[1], picture.shape[2], longest)
    samples = picture.movedim(-1, 1)[:, :3]
    if (h, w) != tuple(samples.shape[-2:]):
        samples = torch.nn.functional.interpolate(samples, size=(h, w), mode="area")
    return samples.movedim(1, -1)


def _target(prompt) -> tuple[int, int] | None:
    """The result's latent size (height, width), from the size Forge's SdConditioning carries (the hires one for the
    hires pass)."""
    width, height = getattr(prompt, "width", None), getattr(prompt, "height", None)
    if not width or not height:
        return None
    return int(height) // geometry.VAE_SCALE, int(width) // geometry.VAE_SCALE


def install(krea2_class) -> None:
    original = krea2_class.get_learned_conditioning

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt):
        options = state.options
        target = _target(prompt)
        if options is None or target is None:
            return original(self, prompt)

        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        engine = self.text_processing_engine_qwen
        pictures = [*self.ref_latents]
        if self.ini_latent is not None:
            pictures.insert(0, self.ini_latent)

        if prompt.is_negative_prompt:
            # Forge encodes the negative first, before the positive takes img2img's picture; CFG 1 skips it. Later in
            # the generation (hires fix) the picture is gone from Forge, but not from the state
            if not state.new_generation and state.pictures:
                pictures = state.pictures
            if options.ground_negative and pictures:
                return engine(prompt, images=[_grounding(p, options.grounding_px) for p in pictures])
            return engine(prompt)

        first = state.new_generation
        self.ini_latent = None
        pictures = state.references_for(pictures)
        if not pictures:
            dynamic_args.ref_latents.clear()
            return engine(prompt)

        lat_h, lat_w = target
        latents = [_encode(self, geometry.fit_pixels(p, lat_h, lat_w, options.fit_mode)) for p in pictures]
        if first or state.owner is not dynamic_args.ref_latents or not dynamic_args.ref_latents:
            dynamic_args.ref_latents = list(latents)
        state.store(target, ReferenceSet(latents, options.fit_mode == geometry.FIT), dynamic_args.ref_latents)
        return engine(prompt, images=[_grounding(p, options.grounding_px) for p in pictures])

    get_learned_conditioning.__wrapped__ = original
    krea2_class.get_learned_conditioning = get_learned_conditioning
