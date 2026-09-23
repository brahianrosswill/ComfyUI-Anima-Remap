"""
Anima VACE ControlNet Remap.

Makes an Anima VACE ControlNet (e.g. TaihoC/Anima-ControlNet-VACE-Depth,
khanghy1000/Anima-ControlNet-VACE-Canny) inject at the right blocks of a 40- or
52-block Anima model.

How a VACE ControlNet uses the host model
-----------------------------------------
The control branch is its own small network (a few copied Anima blocks) that
computes one "hint" per control block. Each hint is ADDED to the output of one
specific block of the host DiT via a forward hook. Those target block indices
come from the checkpoint's metadata -- for the current releases
range(0, 28, 7) = [0, 7, 14, 21], i.e. positions in the 28-block base.

On a 40/52-block model the loader hooks the same indices, so it runs without any
error while injecting at the wrong physical depth (block 7 of a 40-block model
is not what block 7 of the base was). This node rewrites only those target
indices with the same expand manifests the LoRA nodes use; the control branch
itself, its weights and everything the ControlNet computes are untouched.

Nothing is hardcoded: the target indices are read from the loaded ControlNet, so
a future VACE model trained with a different block count or spacing is handled
the same way.

Relationship to ComfyUI-Advanced-ControlNet
-------------------------------------------
The VACE loader lives in a GPL-3.0 fork of ComfyUI-Advanced-ControlNet (the
`fix/anima-vace-hardening` branch of PineCookie/ComfyUI-Advanced-ControlNet).
This module does not import or copy any of that code. It only receives the
CONTROL_NET object that fork produces and adjusts two plain attributes on a
copy of it, so this package stays MIT and still loads normally when the fork
isn't installed -- the dependency only exists when this node is actually used.
"""

import copy
import logging

from .anima_common import (
    AUTO_MANIFEST_LABEL,
    build_base_to_target,
    get_model_block_count,
    list_manifest_choices,
    resolve_manifest,
    resolve_manifest_filename,
)

logger = logging.getLogger("AnimaRemap")


def _find_vace(control_net):
    """
    The VACE model inside the fork's AnimaVACEAdvanced object, or None.

    Duck-typed on purpose (no import of the fork): anything carrying a `vace`
    attribute with `control_layers` / `control_layers_mapping` is the object
    this node knows how to adjust.
    """
    vace = getattr(control_net, "vace", None)
    if vace is None:
        return None
    if not hasattr(vace, "control_layers") or not hasattr(vace, "control_layers_mapping"):
        return None
    return vace


def _host_block_count(model):
    """Exact block count of the connected model; state-dict probing only as a fallback."""
    try:
        return len(model.model.diffusion_model.blocks)
    except Exception:
        return get_model_block_count(model)


class AnimaVACEControlNetRemap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "model": ("MODEL",),
                "auto_remap": ("BOOLEAN", {"default": True}),
                "manifest": (list_manifest_choices(),),
            }
        }

    RETURN_TYPES = ("CONTROL_NET", "STRING")
    RETURN_NAMES = ("control_net", "remap_info")
    FUNCTION = "remap"
    CATEGORY = "loaders/anima"

    def remap(self, control_net, model, auto_remap=True, manifest=AUTO_MANIFEST_LABEL):
        vace = _find_vace(control_net)
        if vace is None:
            raise RuntimeError(
                "This is not an Anima VACE ControlNet. Load it with "
                "'Load Advanced ControlNet Model' from the ComfyUI-Advanced-ControlNet "
                "fork that supports Anima VACE (PineCookie, branch fix/anima-vace-hardening)."
            )

        layers = list(vace.control_layers)
        source_count = getattr(vace, "num_blocks", None)
        host_count = _host_block_count(model)

        if not auto_remap:
            if source_count and host_count and source_count != host_count:
                logger.warning(
                    "VACE ControlNet: auto_remap is OFF but it was trained for %s blocks and the "
                    "model has %s -- hints go to blocks %s, which do not correspond.",
                    source_count, host_count, layers,
                )
                return (control_net,
                        f"auto_remap OFF: {source_count}-block ControlNet on a {host_count}-block "
                        f"model, injecting at {layers} AS-IS (blocks do not correspond)")
            return (control_net, f"applied as-is (auto_remap OFF), injecting at {layers}")

        if source_count is None or host_count is None or source_count == host_count:
            return (control_net,
                    f"applied as-is (blocks={source_count}, model={host_count}), injecting at {layers}")

        if source_count > host_count:
            # Same stance as the LoRA nodes: putting a larger-architecture adapter on a
            # smaller model is not an expected use, so stop rather than guess.
            raise RuntimeError(
                f"VACE ControlNet was trained for {source_count} blocks but the model has only "
                f"{host_count}. Applying a larger-architecture ControlNet to a smaller model is "
                f"not supported."
            )

        manifest_data = resolve_manifest(manifest, source_count, host_count)
        if manifest_data is None:
            raise RuntimeError(
                f"No manifest covers {source_count} -> {host_count} blocks, so the VACE injection "
                f"points cannot be placed. Stopping instead of injecting at the wrong blocks."
            )
        manifest_name = resolve_manifest_filename(manifest, source_count, host_count)
        base_to_target = build_base_to_target(manifest_data)

        missing = [i for i in layers if i not in base_to_target]
        if missing:
            raise RuntimeError(
                f"VACE injection block(s) {missing} are not in {manifest_name}; cannot remap."
            )

        new_layers = [base_to_target[i] for i in layers]
        new_mapping = {base_to_target[b]: ctrl for b, ctrl in vace.control_layers_mapping.items()}

        # Shallow copy of the nn.Module: its parameter/submodule dicts are the SAME
        # objects as the original's, so every weight (and any device placement ComfyUI
        # does on the original) is shared -- nothing is duplicated in memory. Only the
        # two index attributes below differ. The loader's cached ControlNet is never
        # modified, so switching between a 28- and a 40-block model always starts from
        # the original indices.
        remapped = copy.copy(vace)
        remapped.control_layers = new_layers
        remapped.control_layers_mapping = new_mapping

        c = control_net.copy()
        c.vace = remapped
        if hasattr(c, "vace_model"):
            c.vace_model = remapped

        logger.info(
            "VACE ControlNet: %s->%s via %s, injection blocks %s -> %s",
            source_count, host_count, manifest_name, layers, new_layers,
        )
        return (c, f"{source_count}->{host_count} via {manifest_name}, "
                   f"injection blocks {layers} -> {new_layers}")


NODE_CLASS_MAPPINGS = {
    "AnimaVACEControlNetRemap": AnimaVACEControlNetRemap,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaVACEControlNetRemap": "Anima VACE ControlNet Remap",
}
