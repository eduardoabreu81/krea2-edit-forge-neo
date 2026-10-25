# Krea 2 Edit V2 for Forge Neo
# What the current generation asked for, and the references the conditioning prepared for it. The script sets the
# options before each generation; the conditioning (engine.py) fills the reference sets; the DiT (forward.py) reads
# them. No Forge imports: CPU-testable.

from dataclasses import astuple, dataclass

from .geometry import ALIGNED, FIT


@dataclass(frozen=True)
class Options:
    # the last reference: the person in two-picture workflows, the only picture otherwise
    subject_boost: float = 1.0
    subject_position: str = ALIGNED
    # the references before it: the scene in two-picture workflows
    scene_boost: float = 1.0
    scene_position: str = ALIGNED
    fit_mode: str = FIT
    grounding_px: int = 768
    ground_negative: bool = True

    def signature(self) -> tuple:
        return astuple(self)


@dataclass
class ReferenceSet:
    """The references' latents for one result size, in the order the DiT reads them (img2img's picture first, then
    ImageStitch's). centered: fitted by the fit mode for exactly this size, so they keep their own grid."""

    latents: list
    centered: bool


# the options of the generation running now; None when the extension is off, the model is not Krea 2 or Forge's
# "[Krea2] Enable Reference" is off - Forge's own path then runs untouched
options: Options | None = None
# the options the cached conditioning was made with: when they change, the script clears Forge's prompt cache
last_signature: tuple | None = None
# set by the script at the start of every generation; the first positive conditioning of a generation takes the
# pictures from Forge, later ones (hires fix, further batches) reuse them, as Forge hands img2img's picture only once
new_generation: bool = True
pictures: list = []
# reference sets by result latent size (height, width), and the list Forge's dynamic_args.ref_latents held when they
# were made: Forge empties that list when it loads a model, which retires them
sets: dict[tuple[int, int], ReferenceSet] = {}
owner: list | None = None


def start(new: Options | None) -> bool:
    """A generation begins with these options. True when the cached conditioning was made otherwise and has to go."""
    global options, last_signature, new_generation
    options = new
    new_generation = True
    signature = None if new is None else new.signature()
    changed = signature != last_signature
    last_signature = signature
    if changed:
        sets.clear()
    return changed


def references_for(pictures_now: list) -> list:
    """The pictures this conditioning edits: Forge's on the first call of a generation, the same ones after."""
    global new_generation, pictures
    if new_generation or not pictures:
        pictures = list(pictures_now)
        new_generation = False
        sets.clear()
    return pictures


def store(size: tuple[int, int], reference_set: ReferenceSet, forge_list: list) -> None:
    global owner
    if owner is not forge_list:
        sets.clear()
        owner = forge_list
    sets[size] = reference_set


def lookup(size: tuple[int, int], forge_list: list) -> tuple[ReferenceSet | None, bool]:
    """The references for a result of this latent size, and whether they were made for it. A size never conditioned
    (a hires pass whose conditioning came from the cache) gets the first set, to be resized as Forge does."""
    if forge_list is not owner or not forge_list or not sets:
        return None, False
    if size in sets:
        return sets[size], True
    return next(iter(sets.values())), False


def reset() -> None:
    global options, last_signature, new_generation, pictures, owner
    options, last_signature, new_generation, pictures, owner = None, None, True, [], None
    sets.clear()
