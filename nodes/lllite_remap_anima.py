# ---------------------------------------------------------------------------
# Derived from nodes.py of kohya-ss/ComfyUI-Anima-LLLite (Apache License 2.0).
# See lllite_vendor/LICENSE-Apache-2.0 for the full license text.
#
# Because it contains upstream's code, this file is distributed under the
# Apache License 2.0 -- NOT under this repository's MIT license.
# The modifications relative to upstream are listed in the module docstring
# below (see "Changes relative to upstream").
# ---------------------------------------------------------------------------

"""ComfyUI node for Anima ControlNet-LLLite.

Single LoRA-style node: takes a MODEL, an LLLite weights file, a control IMAGE
and a strength; returns the patched MODEL. Integration is done via
``set_model_unet_function_wrapper`` so the LLLite contribution is fully scoped
to this model clone — no global monkey-patching that could leak into other
samplers in the same workflow.

Because ``model_function_wrapper`` is a single-slot field on ``model_options``,
cascading two wrapper-installing nodes would normally cause the outer one to
silently overwrite the inner one. The node captures any pre-existing wrapper
before cloning and delegates to it from inside its own wrapper, so multiple
Anima-LLLite nodes (and other well-behaved wrapper nodes) can be stacked. The
``preserve_wrapper`` toggle (default on) controls this delegation, mirroring
``ChromaRadianceOptions``.

-----------------------------------------------------------------------------
Forked from kohya-ss/ComfyUI-Anima-LLLite (Apache License 2.0); see
lllite_vendor/LICENSE-Apache-2.0. Upstream explicitly invites community nodes
built on that codebase.

Changes relative to upstream's AnimaLLLiteApply_sdscripts:
  * Block indices in the checkpoint's keys are remapped to the connected model's
    architecture (28 / 40 / 52) before loading, using the same expand manifests
    the LoRA nodes use -- so one LLLite checkpoint works on every Anima
    generation. See lllite_block_remap.py.
  * New `auto_remap` and `manifest` inputs, and a `remap_info` STRING output
    reporting what was detected and done.
  * Optional `extend_to_new_layers` / `extend_strength`: when expanding, give the
    newly-inserted blocks a copy of their predecessor's module instead of leaving
    them as no-ops. Off by default.
  * Node ID renamed so it can coexist with both upstream's node and ComfyUI's
    built-in AnimaLLLiteApply.

Everything else -- conditioning preprocessing, the inpaint (4ch) path, step-range
gating and the wrapper-delegation behaviour -- is upstream's, unchanged.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

import folder_paths

from .lllite_vendor.control_net_lllite_anima import (
    ASPP_DEFAULT_DILATIONS,
    ControlNetLLLiteDiT,
    read_lllite_metadata,
)
# Private in upstream, but importing it (rather than adding another public
# wrapper there) keeps the vendored file as close to upstream as possible.
from .lllite_vendor.control_net_lllite_anima import (
    _from_saved_state_dict,
    _INTERNAL_MODULES_PREFIX,
)
from .anima_common import (
    AUTO_MANIFEST_LABEL,
    upcast_float8_state_dict,
    get_model_block_count,
    list_manifest_choices,
)
from .lllite_block_remap import (
    build_lllite_block_map,
    build_lllite_extension_sources,
    extend_lllite_state_dict,
    get_host_block_count,
    get_lllite_block_count,
    remap_lllite_state_dict,
)

logger = logging.getLogger("AnimaRemap")


def _get_inner_dit(model) -> torch.nn.Module:
    """Reach the underlying Anima DiT (nn.Module) from a ComfyUI ModelPatcher."""
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError("Input MODEL has no .model attribute (not a ModelPatcher?)")
    dit = getattr(inner, "diffusion_model", None)
    if dit is None:
        raise RuntimeError("MODEL.model has no .diffusion_model — not a UNet/DiT model?")
    return dit


def _target_cond_hw(latent_h: int, latent_w: int, patch_spatial: int = 2) -> tuple[int, int]:
    """Return the (H, W) the cond image / mask must be resized to.

    The LLLite ``conditioning1`` Conv has stride 16, so the cond image must be
    sized to ``latent_HW * 8`` in input pixel space (= ``token_HW * 16`` after
    DiT patchify with patch_spatial=2). The DiT internally pads the latent up
    to a multiple of ``patch_spatial`` (see ``MiniTrainDIT.forward`` →
    ``pad_to_patch_size``), so we mirror that rounding here — otherwise odd
    latent dims (e.g. 1032 px → 129 latent) yield a token-count mismatch that
    silently bypasses every LLLite module.
    """
    padded_h = ((latent_h + patch_spatial - 1) // patch_spatial) * patch_spatial
    padded_w = ((latent_w + patch_spatial - 1) // patch_spatial) * patch_spatial
    return padded_h * 8, padded_w * 8


def _prepare_cond_image(image: torch.Tensor, latent_h: int, latent_w: int,
                        device: torch.device, dtype: torch.dtype,
                        patch_spatial: int = 2) -> torch.Tensor:
    """ComfyUI IMAGE (B,H,W,3) in [0,1] → (1,3,H*8,W*8) in [-1,1]."""
    if image.ndim == 4 and image.shape[-1] == 3:
        # (B, H, W, 3) -> (B, 3, H, W)
        img = image.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Unexpected cond image shape: {tuple(image.shape)} (expected B,H,W,3)")

    img = img[:1]  # use first frame only
    target_h, target_w = _target_cond_hw(latent_h, latent_w, patch_spatial)
    if img.shape[-2] != target_h or img.shape[-1] != target_w:
        img = F.interpolate(img, size=(target_h, target_w), mode="bicubic", align_corners=False)
        img = img.clamp(0.0, 1.0)
    img = img * 2.0 - 1.0
    return img.to(device=device, dtype=dtype)


def _prepare_mask(mask: torch.Tensor, latent_h: int, latent_w: int,
                  device: torch.device, dtype: torch.dtype,
                  patch_spatial: int = 2) -> torch.Tensor:
    """ComfyUI MASK (B,H,W) in [0,1] → (1,1,H*8,W*8) binarized at 0.5.

    Returns the mask in ``{0.0, 1.0}`` (1 = inpaint area, 0 = keep). The caller
    is responsible for the ``*2-1`` rescale before concat with RGB.
    """
    if mask.ndim == 3:
        m = mask.unsqueeze(1)              # (B, 1, H, W)
    elif mask.ndim == 4 and mask.shape[1] == 1:
        m = mask
    else:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)} (expected B,H,W or B,1,H,W)")

    m = m[:1]
    target_h, target_w = _target_cond_hw(latent_h, latent_w, patch_spatial)
    if m.shape[-2] != target_h or m.shape[-1] != target_w:
        m = F.interpolate(m.float(), size=(target_h, target_w), mode="nearest")
    m = (m >= 0.5).to(dtype=dtype)
    return m.to(device=device)


def _build_inpaint_cond_image(rgb_pm1: torch.Tensor, mask01: torch.Tensor,
                              masked_input: bool) -> torch.Tensor:
    """rgb_pm1: (1,3,H,W) in [-1,1], mask01: (1,1,H,W) in {0,1}. Returns (1,4,H,W).

    Mirrors ``_build_inpaint_cond_image`` in the sd-scripts training / inference
    code: the mask channel is rescaled to ``[-1, +1]`` (matches the RGB range),
    and if ``masked_input`` is set the RGB is zeroed where ``mask >= 0.5``.
    """
    if masked_input:
        keep = (mask01 < 0.5).to(rgb_pm1.dtype)
        rgb_pm1 = rgb_pm1 * keep
    mask_pm1 = mask01.to(rgb_pm1.dtype) * 2.0 - 1.0
    return torch.cat([rgb_pm1, mask_pm1], dim=1)


def _load_with_remap(lllite, weights_path, lllite_name, model, auto_remap, manifest,
                     extend_to_new_layers=False, extend_strength=0.5):
    """
    Load the checkpoint, remapping its block indices to the connected model first.

    Returns the remap_info string.
    """
    if os.path.splitext(weights_path)[1] == ".safetensors":
        from safetensors.torch import load_file
        weights_sd = load_file(weights_path)
    else:
        weights_sd = torch.load(weights_path, map_location="cpu")

    # Some LLLite files are distributed in fp8. PyTorch has almost no CPU arithmetic
    # for fp8, so the remap/extension/zero-fill steps below would fail on them.
    # load_state_dict() casts into the module's own dtype anyway, so upcasting the
    # (small) checkpoint here changes nothing about the result.
    weights_sd = upcast_float8_state_dict(weights_sd)

    # Same guard upstream's load_lllite_weights() applies: v1-format files use the
    # internal module names directly and would load into the wrong places.
    if any(k.startswith(_INTERNAL_MODULES_PREFIX) for k in weights_sd):
        raise RuntimeError(
            f"weights at {weights_path} appear to be in a legacy ControlNet-LLLite "
            f"weight format (keys starting with '{_INTERNAL_MODULES_PREFIX}'). "
            f"Re-train with the current codebase."
        )

    source_count = get_lllite_block_count(weights_sd.keys())
    # The module list is ground truth for what is about to be loaded; the model
    # state-dict probe is only a fallback.
    target_count = get_host_block_count(lllite) or get_model_block_count(model)
    info = []

    if not auto_remap:
        if source_count is not None and target_count is not None and source_count != target_count:
            logger.warning(
                "'%s': auto_remap is OFF but the checkpoint is %s-block and the model is "
                "%s-block -- its modules will land on the wrong physical blocks.",
                lllite_name, source_count, target_count,
            )
            info.append(
                f"{lllite_name}: auto_remap OFF, {source_count}-block weights applied to a "
                f"{target_count}-block model AS-IS (blocks do not correspond)"
            )
        else:
            info.append(f"{lllite_name}: applied as-is (auto_remap OFF)")
        block_map = None
    elif source_count is None or target_count is None or source_count == target_count:
        info.append(
            f"{lllite_name}: applied as-is (blocks={source_count}, model={target_count})"
        )
        block_map = None
    else:
        block_map, manifest_name = build_lllite_block_map(manifest, source_count, target_count)
        if block_map is None:
            raise RuntimeError(
                f"'{lllite_name}': no manifest covers {source_count} -> {target_count} blocks. "
                f"Applying it unchanged would put every module on the wrong block, so loading "
                f"is stopped instead."
            )
        weights_sd, renamed, dropped = remap_lllite_state_dict(weights_sd, block_map)
        logger.info(
            "'%s': %s->%s via %s, %d module tensors remapped, %d dropped",
            lllite_name, source_count, target_count, manifest_name, renamed, dropped,
        )
        info.append(
            f"{lllite_name}: {source_count}->{target_count} via {manifest_name}, "
            f"{renamed} tensors remapped, {dropped} dropped"
        )
        if extend_to_new_layers and source_count < target_count:
            sources = build_lllite_extension_sources(manifest, source_count, target_count)
            weights_sd, extended = extend_lllite_state_dict(weights_sd, sources, extend_strength)
            logger.info(
                "'%s': extended %d module(s) onto new blocks (copy from predecessor, "
                "strength=%s)", lllite_name, extended, extend_strength,
            )
            info.append(
                f"{lllite_name}: extended {extended} module(s) onto new blocks "
                f"(copy from predecessor, strength={extend_strength})"
            )
        uncovered = len(lllite.lllite_modules) - len(
            {k.split(".")[0] for k in weights_sd if k.startswith("lllite_dit_blocks_")}
        )
        if uncovered > 0:
            info.append(
                f"{lllite_name}: {uncovered} module(s) on newly-inserted blocks left "
                f"untrained (no-op)"
            )

    converted = _from_saved_state_dict(lllite, weights_sd)
    result = lllite.load_state_dict(converted, strict=False)
    logger.info("loaded LLLite weights from %s: %s", weights_path, result)
    return "\n".join(info)


class AnimaLLLiteRemapApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lllite_name": (folder_paths.get_filename_list("controlnet"),),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "auto_remap": ("BOOLEAN", {"default": True}),
                "manifest": (list_manifest_choices(),),
                "extend_to_new_layers": ("BOOLEAN", {"default": False}),
                "extend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "preserve_wrapper": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # Required when the loaded weights are 4ch (inpaint). White = inpaint area,
                # black = keep. Mismatch with the weights' cond_in_channels is reported below.
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "remap_info")
    FUNCTION = "apply"
    CATEGORY = "loaders/anima"

    def apply(self, model, lllite_name, image, strength, start_percent, end_percent,
              auto_remap=True, manifest=AUTO_MANIFEST_LABEL,
              extend_to_new_layers=False, extend_strength=0.5,
              preserve_wrapper=True, mask=None):
        weights_path = folder_paths.get_full_path("controlnet", lllite_name)
        if weights_path is None or not os.path.isfile(weights_path):
            raise FileNotFoundError(f"LLLite weights not found: {lllite_name}")

        # Architecture is fully determined by the trained weights — read everything
        # from metadata rather than exposing knobs that would just cause load errors.
        meta = read_lllite_metadata(weights_path)
        ce_dim = int(meta.get("lllite.cond_emb_dim", 32))
        m_dim = int(meta.get("lllite.mlp_dim", 64))
        # v2 records the canonical atomic form under lllite.target_atomics; fall back
        # to the legacy preset key, then to the v1 default.
        tl = meta.get("lllite.target_atomics", meta.get("lllite.target_layers", "self_attn_q"))
        cond_dim = int(meta.get("lllite.cond_dim", 64))
        cond_resblocks = int(meta.get("lllite.cond_resblocks", 1))
        use_aspp = str(meta.get("lllite.use_aspp", "false")).lower() == "true"
        aspp_dilations_meta = meta.get("lllite.aspp_dilations")
        if use_aspp and aspp_dilations_meta:
            aspp_dilations = tuple(int(d) for d in aspp_dilations_meta.split(",") if d.strip())
        else:
            aspp_dilations = ASPP_DEFAULT_DILATIONS
        cond_in_channels = int(meta.get("lllite.cond_in_channels", 3))
        inpaint_masked_input = str(meta.get("lllite.inpaint_masked_input", "false")).lower() == "true"

        # Mask / cond_in_channels consistency: 4ch weights need a MASK, 3ch weights ignore it.
        if cond_in_channels == 4 and mask is None:
            raise ValueError(
                f"LLLite weights '{lllite_name}' were trained with cond_in_channels=4 "
                f"(inpaint mode) and require a MASK input. Connect a MASK to the node "
                f"(white = inpaint area, black = keep)."
            )
        if cond_in_channels != 4 and mask is not None:
            logger.warning(
                "LLLite weights '%s' are %dch; the provided MASK input will be ignored.",
                lllite_name, cond_in_channels,
            )
            mask = None

        dit = _get_inner_dit(model)
        patch_spatial = int(getattr(dit, "patch_spatial", 2))
        lllite = ControlNetLLLiteDiT(
            dit,
            cond_emb_dim=ce_dim,
            mlp_dim=m_dim,
            target_layers=tl,
            multiplier=strength,
            cond_dim=cond_dim,
            cond_resblocks=cond_resblocks,
            use_aspp=use_aspp,
            aspp_dilations=aspp_dilations,
            cond_in_channels=cond_in_channels,
            inpaint_masked_input=inpaint_masked_input,
        )
        remap_info = _load_with_remap(
            lllite, weights_path, lllite_name, model, auto_remap, manifest,
            extend_to_new_layers, extend_strength,
        )
        lllite.eval().requires_grad_(False)

        # Convert percent range -> sigma range (start_percent=0 → sigma_max).
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))

        # Capture image / mask tensors (cloned to detach from any upstream caching)
        src_image = image.detach().clone()
        src_mask = mask.detach().clone() if mask is not None else None
        is_inpaint = cond_in_channels == 4

        # Cache for the per-resolution preprocessed cond image (avoids repeat resize)
        cache = {"cond_image_pp": None, "key": None, "lllite_loaded_to": None}

        # Capture any previously-installed wrapper BEFORE we clone — model_options
        # has a single "model_function_wrapper" slot, so without delegation a second
        # wrapper-installing node would silently no-op the first. Mirrors the
        # ChromaRadianceOptions pattern in comfy_extras/nodes_chroma_radiance.py.
        old_wrapper = model.model_options.get("model_function_wrapper")

        def _call_next(apply_model, input_x, timestep, c):
            if preserve_wrapper and old_wrapper is not None:
                return old_wrapper(apply_model, {"input": input_x, "timestep": timestep, "c": c})
            return apply_model(input_x, timestep, **c)

        def wrapper(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            # Step-range gate: skip LLLite entirely when current sigma is outside
            # [sigma_end, sigma_start]. percent_to_sigma maps 0.0 → sigma_max,
            # 1.0 → sigma_min, so the active window is sigma_end <= sigma <= sigma_start.
            sigma = float(timestep.max().item())
            if not (sigma_end <= sigma <= sigma_start):
                return _call_next(apply_model, input_x, timestep, c)

            # Anima latent shape: (B, C, T, H, W) — take spatial dims from the tail.
            latent_h, latent_w = int(input_x.shape[-2]), int(input_x.shape[-1])
            device = input_x.device
            dtype = input_x.dtype

            # Move LLLite to the runtime device/dtype lazily.
            tag = (device, dtype)
            if cache["lllite_loaded_to"] != tag:
                lllite.to(device=device, dtype=dtype)
                cache["lllite_loaded_to"] = tag
                cache["cond_image_pp"] = None  # invalidate

            key = (latent_h, latent_w, device, dtype)
            if cache["key"] != key or cache["cond_image_pp"] is None:
                rgb = _prepare_cond_image(
                    src_image, latent_h, latent_w, device, dtype, patch_spatial
                )
                if is_inpaint:
                    mk = _prepare_mask(
                        src_mask, latent_h, latent_w, device, dtype, patch_spatial
                    )
                    cache["cond_image_pp"] = _build_inpaint_cond_image(
                        rgb, mk, inpaint_masked_input
                    )
                else:
                    cache["cond_image_pp"] = rgb
                cache["key"] = key

            lllite.set_multiplier(strength)
            lllite.set_cond_image(cache["cond_image_pp"])
            lllite.apply_to()
            try:
                return _call_next(apply_model, input_x, timestep, c)
            finally:
                lllite.restore()
                lllite.clear_cond_image()

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        return (m, remap_info)


NODE_CLASS_MAPPINGS = {
    "AnimaLLLiteRemapApply": AnimaLLLiteRemapApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLLLiteRemapApply": "Apply Anima ControlNet-LLLite (Auto Remap)",
}
