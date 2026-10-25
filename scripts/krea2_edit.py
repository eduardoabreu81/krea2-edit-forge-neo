# Krea 2 Edit V2 for Forge Neo
# Entry point: Forge Neo imports every scripts/*.py at startup, before the UI and before any model loads. The hooks go
# in at module scope; any failure leaves Forge Neo unchanged and says so in the console. The accordion below sets the
# V2 controls for each generation.

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENABLED = False

try:
    from forge_krea2edit import compat

    _problems = compat.check()
    if _problems:
        compat.report(_problems)
    elif (_native := compat.native_support()) is not None:
        compat.report_native(_native)
    else:
        from forge_krea2edit import patches

        patches.apply()
        ENABLED = True
except Exception:
    print("[Krea2 Edit V2] failed to enable, Forge Neo is unchanged:")
    traceback.print_exc()

if ENABLED:
    import gradio as gr
    from backend.diffusion_engine.krea import Krea2
    from modules import scripts
    from modules.shared import opts
    from modules.ui_components import FormRow, InputAccordion

    from forge_krea2edit import __version__, state
    from forge_krea2edit.geometry import ALIGNED, FIT, FIT_MODES, POSITIONS
    from forge_krea2edit.state import Options

    KEY = "Krea2 Edit V2"
    DEFAULTS = Options()

    INFO = """
    Controls of the <b>Krea 2 Identity Edit V2</b> and <b>General Edit V2</b> LoRAs, on top of Forge Neo's own Krea 2
    edit: turn on <i>Settings → Stable Diffusion → [Krea2] Enable Reference</i> and load one of the LoRAs.<br>
    The pictures to edit are img2img's picture, then ImageStitch's, in that order; the <b>last</b> one is the
    <b>subject</b> (the only picture when there is one), the ones before it the <b>scene</b>.
    """

    def stitched(p) -> bool:
        """Whether ImageStitch adds pictures to p."""
        for script in getattr(p.scripts, "alwayson_scripts", None) or ():
            if script.title() == "ImageStitch Integrated":
                try:
                    enable, references, *_ = p.script_args[script.args_from:script.args_to]
                    return bool(enable and references)
                except Exception:  # arguments not shaped as its UI makes them
                    return False
        return False

    def infotext(options: Options) -> dict:
        params = {KEY: options.fit_mode}
        if options.subject_boost != DEFAULTS.subject_boost:
            params[f"{KEY} boost"] = f"{options.subject_boost:g}"
        if options.subject_position != DEFAULTS.subject_position:
            params[f"{KEY} placement"] = options.subject_position
        if options.scene_boost != DEFAULTS.scene_boost:
            params[f"{KEY} scene boost"] = f"{options.scene_boost:g}"
        if options.scene_position != DEFAULTS.scene_position:
            params[f"{KEY} scene placement"] = options.scene_position
        if options.grounding_px != DEFAULTS.grounding_px:
            params[f"{KEY} grounding"] = options.grounding_px
        if options.ground_negative != DEFAULTS.ground_negative:
            params[f"{KEY} negative grounding"] = options.ground_negative
        return params

    def pasted(key: str, default):
        return lambda d: d.get(key, default) if KEY in d else None

    class Krea2EditV2(scripts.Script):
        def title(self):
            return "Krea 2 Edit V2"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            eid = lambda name: f"krea2_edit_v2_{'img2img' if is_img2img else 'txt2img'}_{name}"  # noqa: E731
            with InputAccordion(False, label=f"{self.title()} {__version__}", elem_id=eid("enabled")) as enabled:
                gr.HTML(INFO)
                with FormRow():
                    subject_boost = gr.Slider(0.0, 10.0, value=DEFAULTS.subject_boost, step=0.05, label="Subject boost",
                                              info="how hard the result follows the subject picture; 1 = off. "
                                                   "Identity V2: 2 (2-3 with a scene), General V2: 4",
                                              elem_id=eid("subject_boost"))
                    subject_position = gr.Radio(list(POSITIONS), value=ALIGNED, label="Subject placement",
                                                info="beside: the person can be restaged more freely",
                                                elem_id=eid("subject_position"))
                with FormRow():
                    scene_boost = gr.Slider(0.0, 10.0, value=DEFAULTS.scene_boost, step=0.05, label="Scene boost",
                                            info="the same for the pictures before the last; 1 = off",
                                            elem_id=eid("scene_boost"))
                    scene_position = gr.Radio(list(POSITIONS), value=ALIGNED, label="Scene placement",
                                              info="to put a person into a scene: scene aligned, subject beside",
                                              elem_id=eid("scene_position"))
                with FormRow():
                    fit_mode = gr.Radio(list(FIT_MODES), value=FIT, label="Fit",
                                        info="how a picture of other proportions meets the result; crop (legacy) "
                                             "for the v1 / v1.1 LoRAs",
                                        elem_id=eid("fit_mode"))
                    grounding_px = gr.Slider(0, 2048, value=DEFAULTS.grounding_px, step=64, label="Grounding size",
                                             info="longest side the text encoder sees a picture at; 0 = as it is",
                                             elem_id=eid("grounding_px"))
                    ground_negative = gr.Checkbox(DEFAULTS.ground_negative, label="Ground the negative prompt",
                                                  info="show the pictures with the negative prompt too (CFG above 1)",
                                                  elem_id=eid("ground_negative"))

            self.infotext_fields = [
                (enabled, lambda d: KEY in d),
                (subject_boost, pasted(f"{KEY} boost", DEFAULTS.subject_boost)),
                (subject_position, pasted(f"{KEY} placement", DEFAULTS.subject_position)),
                (scene_boost, pasted(f"{KEY} scene boost", DEFAULTS.scene_boost)),
                (scene_position, pasted(f"{KEY} scene placement", DEFAULTS.scene_position)),
                (fit_mode, pasted(KEY, DEFAULTS.fit_mode)),
                (grounding_px, pasted(f"{KEY} grounding", DEFAULTS.grounding_px)),
                (ground_negative, pasted(f"{KEY} negative grounding", DEFAULTS.ground_negative)),
            ]
            return [enabled, subject_boost, subject_position, scene_boost, scene_position, fit_mode, grounding_px,
                    ground_negative]

        def process(self, p, enabled, subject_boost, subject_position, scene_boost, scene_position, fit_mode,
                    grounding_px, ground_negative, *args):
            options = None
            if enabled and isinstance(p.sd_model, Krea2):
                if not getattr(opts, "krea2_do_reference", False):
                    print("[Krea2 Edit V2] off for this generation: turn on Settings -> Stable Diffusion -> "
                          "[Krea2] Enable Reference")
                elif self.is_img2img or stitched(p):
                    options = Options(float(subject_boost), str(subject_position), float(scene_boost),
                                      str(scene_position), str(fit_mode), int(grounding_px), bool(ground_negative))
            if state.start(options):
                # the cached conditioning holds references made another way
                p.clear_prompt_cache()
            if options is not None:
                p.extra_generation_params.update(infotext(options))

        def postprocess(self, p, processed, *args):
            # nothing outside a generation goes through the V2 path; the reference sets stay for a cached conditioning
            state.options = None
