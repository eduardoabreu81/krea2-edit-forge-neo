# Krea 2 Edit V2 for Forge Neo
# Checks that the installed Forge Neo has the Krea 2 edit path the extension builds on, before anything is patched,
# and that it does not have the V2 controls by itself. Same mechanism as the FIBO 2 extension's compat.py.

import importlib
import inspect

# what the extension imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.args": ("dynamic_args",),
    "backend.memory_management": ("load_model_gpu",),
    "backend.misc.image_resize": ("adaptive_resize",),
    "backend.nn.flux": ("timestep_embedding",),
    "backend.nn.krea": ("SingleStreamDiT",),
    "backend.utils": ("pad_to_patch_size",),
    "backend.diffusion_engine.krea": ("Krea2",),
}

# the pieces of Forge's Krea 2 edit path the hooks stand in for: an edit forward reading dynamic_args.ref_latents,
# and a conditioning that keeps img2img's picture in ini_latent and ImageStitch's in ref_latents, as pixels
FORWARD_MARKERS = ("dynamic_args.ref_latents", "self.first", "self.txtfusion", "self.txtmlp", "self.pe_embedder")
CONDITIONING_MARKERS = ("self.ini_latent", "self.ref_latents", "text_processing_engine_qwen", "krea2_do_reference")
DIT_ATTRIBUTES = ("first", "blocks", "tmlp", "tproj", "txtfusion", "txtmlp", "last", "pe_embedder")
# words of the V2 controls, were Forge's own forward to gain them
NATIVE_MARKERS = ("ref_boost", "beside")


def _source(function) -> str:
    return inspect.getsource(inspect.unwrap(function))


def native_support() -> str | None:
    """What Forge Neo's own Krea 2 forward says about the V2 controls, when it has them."""
    from backend.nn import krea

    source = _source(krea.SingleStreamDiT.forward)
    found = [m for m in NATIVE_MARKERS if m in source]
    return ", ".join(found) if found else None


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")
    if problems:
        return problems

    from backend.args import dynamic_args
    from backend.diffusion_engine.krea import Krea2
    from backend.nn import krea

    if not isinstance(getattr(dynamic_args, "ref_latents", None), list):
        problems.append("backend.args.dynamic_args has no ref_latents list")

    source = _source(krea.SingleStreamDiT.forward)
    missing = [m for m in FORWARD_MARKERS if m not in source]
    if missing:
        problems.append(f"the Krea 2 DiT forward does not read {', '.join(missing)}")
    init = _source(krea.SingleStreamDiT.__init__)
    missing = [a for a in DIT_ATTRIBUTES if f"self.{a} =" not in init]
    if missing:
        problems.append(f"the Krea 2 DiT has no {', '.join(missing)}")

    for name in ("get_learned_conditioning", "encode_first_stage"):
        if not hasattr(Krea2, name):
            problems.append(f"the Krea 2 engine has no {name}")
    if not problems:
        source = _source(Krea2.get_learned_conditioning) + _source(Krea2.encode_first_stage)
        missing = [m for m in CONDITIONING_MARKERS if m not in source]
        if missing:
            problems.append(f"the Krea 2 engine's edit conditioning does not use {', '.join(missing)}")

    return problems


def report(problems: list[str]) -> None:
    print("[Krea2 Edit V2] this Forge Neo is too old or too new for the extension, so it stays disabled "
          "(Forge Neo's own Krea 2 edit keeps working as usual):")
    for problem in problems:
        print(f"          {problem}")


def report_native(found: str) -> None:
    print(f"[Krea2 Edit V2] this Forge Neo's Krea 2 forward seems to have the V2 controls by itself ({found}), so "
          "the extension stays out of its way; you can uninstall it")
