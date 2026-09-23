"""
lora_remap_extended_anima.py

EXPERIMENTAL variant of AnimaLoRARemapTagLoader. Same base remapping
behavior, but generalizes how the LoRA's effect is projected onto
Anima-2.9B's 12 newly-inserted layers: instead of always copying from the
single "source" layer recorded in expand_manifest.json (which is always
the immediately-PRECEDING old layer), this node blends between the
preceding ("front") and following ("back") old layers using `blend_ratio`.

    blended_value = front_layer_value * blend_ratio + back_layer_value * (1 - blend_ratio)

At blend_ratio = 1.0, this is mathematically identical to the original
node's extend_to_new_layers behavior (100% front, 0% back). Lower values
progressively mix in the following layer instead.

This is a SEPARATE node (not a modification of lora_remap_anima.py) so
the original, simpler node keeps working exactly as before.
"""

import logging
import os

import folder_paths
import comfy.utils
import comfy.sd

from .anima_common import (
    computable,
    get_lora_block_count,
    remap_key,
    split_block_key,
    list_manifest_choices,
    resolve_manifest,
    build_base_to_target,
    build_insertion_neighbors,
    get_model_block_count,
    AnimaBlockMismatchError,
    compute_remap_settings_hash,
    peek_lora_block_count,
    resolve_manifest_filename,
)
from .lora_remap_anima import (
    resolve_lora_path,
    save_remapped_lora,
    parse_lora_tags,
    warn_if_legacy_cache,
    lora_lookup_diagnostic,
    _strip_lora_ext,
    _normalize_lookup_name,
)

logger = logging.getLogger("AnimaLoRARemapExtended")

REMAP_SUFFIX_EXT = "_animaremap"


def remap_cache_suffix_ext(target_block_count, settings_hash=None):
    """Same reasoning as remap_cache_suffix() in lora_remap_anima.py, plus the
    settings hash (see compute_remap_settings_hash) so a cache made under one
    manifest/blend_ratio/extend_strength combination is never silently reused
    after those change. Omitting the hash yields the LEGACY suffix, used only to
    detect and warn about pre-hash cache files."""
    if settings_hash:
        return f"{REMAP_SUFFIX_EXT}{target_block_count}_ext_{settings_hash}"
    return f"{REMAP_SUFFIX_EXT}{target_block_count}_ext"


def get_remapped_sibling_path_ext(original_path, target_block_count, settings_hash=None):
    """Path for the cached extended-remap copy: same folder, same extension."""
    base, ext = os.path.splitext(original_path)
    return f"{base}{remap_cache_suffix_ext(target_block_count, settings_hash)}{ext}"


def group_by_base_index(lora_sd):
    """{base_idx: {suffix: (prefix, sep, tensor)}} for every MAIN-block tensor."""
    groups = {}
    for k, v in lora_sd.items():
        parsed = split_block_key(k)
        if parsed is None:
            continue
        prefix, base_idx, suffix, sep = parsed
        groups.setdefault(base_idx, {})[suffix] = (prefix, sep, v)
    return groups


def build_blended_extension(groups, neighbors, blend_ratio):
    """
    For each inserted target layer, blend its front/back neighbor tensors
    (matched by subkey suffix) using blend_ratio. Returns {new_key: tensor}.

    Near either end of the block sequence, an inserted layer may have only
    ONE of the two neighbors (see build_insertion_neighbors). In that case
    there is nothing to blend against, so that sole neighbor is used at full
    strength -- NOT scaled by blend_ratio, which could otherwise zero out
    the whole extension for that layer if blend_ratio favored the missing side.
    """
    extended = {}
    for t_idx, (prev_base, next_base) in neighbors.items():
        prev_group = groups.get(prev_base, {}) if prev_base is not None else {}
        next_group = groups.get(next_base, {}) if next_base is not None else {}
        all_suffixes = set(prev_group) | set(next_group)
        for suffix in all_suffixes:
            prev_entry = prev_group.get(suffix)
            next_entry = next_group.get(suffix)
            if prev_entry and next_entry:
                prefix, sep, v_prev = prev_entry
                _, _, v_next = next_entry
                blended = computable(v_prev) * blend_ratio + computable(v_next) * (1.0 - blend_ratio)
            elif prev_entry:
                # Only the front neighbor exists -- no other side to blend
                # against, so use it at full strength.
                prefix, sep, v_prev = prev_entry
                blended = v_prev
            elif next_entry:
                # Only the back neighbor exists -- same reasoning as above.
                prefix, sep, v_next = next_entry
                blended = v_next
            else:
                continue
            new_key = f"{prefix}{t_idx}{sep}{suffix}"
            extended[new_key] = blended
    return extended


class AnimaLoRARemapExtendedTagLoader:
    """
    EXPERIMENTAL: same as AnimaLoRARemapTagLoader, but the extension onto
    Anima-2.9B's 12 newly-inserted layers uses a continuous front/back
    blend_ratio instead of always copying from the single preceding layer.
    blend_ratio=1.0 reproduces the original node's behavior exactly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        manifests = list_manifest_choices()
        return {
            "required": {
                "model": ("MODEL",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "default_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "weight_multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "auto_remap": ("BOOLEAN", {"default": True}),
                "save_remapped": ("BOOLEAN", {"default": False}),
                "extend_to_new_layers": ("BOOLEAN", {"default": False}),
                "blend_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "extend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "manifest": (manifests,),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "text", "remap_info")
    FUNCTION = "load"
    CATEGORY = "loaders/anima/experimental"

    def load(self, model, text, default_weight, weight_multiplier, auto_remap, save_remapped,
              extend_to_new_layers, blend_ratio, extend_strength, manifest, clip=None):
        tags, stripped_text = parse_lora_tags(text, default_weight)

        out_model = model
        out_clip = clip
        remap_info_lines = []

        # Same for every tag this run; the manifest itself is resolved per
        # tag below (a mixed prompt can reference LoRAs from different
        # Anima generations, each needing a different manifest against the
        # same connected model).
        model_block_count = get_model_block_count(out_model) if auto_remap else None

        for name, w_model, w_clip in tags:
            original_path = resolve_lora_path(name)
            if original_path is None:
                logger.warning(
                    f"LoRA not found in loras folder (tried exact name, common "
                    f"extensions, and a subfolder basename search): {name} -- "
                    f"{lora_lookup_diagnostic(name)}"
                )
                continue

            # The cache is keyed by (name, target_block_count, settings_hash).
            # The hash needs the RESOLVED manifest, which needs this LoRA's own
            # block count -- read from the safetensors header only, so picking
            # the cache filename doesn't cost a full file load.
            cached_path = None
            settings_hash = None
            cache_stem = _strip_lora_ext(os.path.basename(_normalize_lookup_name(original_path)))
            if auto_remap and model_block_count is not None:
                peeked = peek_lora_block_count(original_path)
                manifest_name = resolve_manifest_filename(manifest, peeked, model_block_count)
                if manifest_name:
                    settings_hash = compute_remap_settings_hash(
                        manifest_name, model_block_count,
                        extend_to_new_layers, extend_strength,
                        blend_ratio=blend_ratio, extended_node=True,
                    )
                    # Built from the RESOLVED path, never from the raw tag text.
                    cached_path = resolve_lora_path(
                        f"{cache_stem}{remap_cache_suffix_ext(model_block_count, settings_hash)}"
                    )
                if cached_path is None:
                    warn_if_legacy_cache(cache_stem, model_block_count, remap_cache_suffix_ext)

            if cached_path is not None:
                lora_sd = comfy.utils.load_torch_file(cached_path, safe_load=True)
                logger.info(f"'{name}': using cached extended-remap file {cached_path}")
                remap_info_lines.append(
                    f"{name}: cache hit ({os.path.basename(cached_path)}), target={model_block_count}"
                )
            else:
                lora_sd = comfy.utils.load_torch_file(original_path, safe_load=True)
                lora_block_count = get_lora_block_count(lora_sd.keys())
                needs_remap = (
                    auto_remap
                    and model_block_count is not None
                    and lora_block_count is not None
                    and lora_block_count < model_block_count
                )

                if needs_remap:
                    manifest_used = resolve_manifest_filename(manifest, lora_block_count, model_block_count)
                    manifest_data = resolve_manifest(manifest, lora_block_count, model_block_count)
                    base_to_target = build_base_to_target(manifest_data) if manifest_data else {}
                    neighbors = build_insertion_neighbors(manifest_data) if manifest_data else {}

                    if base_to_target:
                        remapped = {}
                        dropped = 0
                        for k, v in lora_sd.items():
                            new_k, _ = remap_key(k, base_to_target)
                            if new_k is None:
                                dropped += 1
                                continue
                            remapped[new_k] = v
                        logger.info(
                            f"'{name}': remapped {lora_block_count}->{model_block_count} blocks "
                            f"({len(remapped)} tensors kept, {dropped} dropped as "
                            f"newly-inserted layers with no old counterpart)"
                        )
                        remap_info_lines.append(
                            f"{name}: {lora_block_count}->{model_block_count} via "
                            f"{manifest_used}, {len(remapped)} keys kept, {dropped} dropped"
                        )

                        if extend_to_new_layers and neighbors:
                            groups = group_by_base_index(lora_sd)
                            extension = build_blended_extension(groups, neighbors, blend_ratio)
                            for k, v in extension.items():
                                remapped[k] = computable(v) * extend_strength
                            logger.info(
                                f"'{name}': [experimental] extended {len(extension)} tensors onto "
                                f"newly-inserted layers via front/back blend "
                                f"(blend_ratio={blend_ratio}, extend_strength={extend_strength})"
                            )
                            remap_info_lines.append(
                                f"{name}: extended {len(extension)} tensors (blend_ratio={blend_ratio}, "
                                f"strength={extend_strength})"
                            )

                        lora_sd = remapped

                        if save_remapped:
                            remapped_path = get_remapped_sibling_path_ext(
                                original_path, model_block_count,
                                settings_hash or compute_remap_settings_hash(
                                    manifest_used, model_block_count,
                                    extend_to_new_layers, extend_strength,
                                    blend_ratio=blend_ratio, extended_node=True,
                                ),
                            )
                            if os.path.exists(remapped_path):
                                logger.info(f"'{name}': extended-remap cache already exists, skipping save")
                            else:
                                try:
                                    save_remapped_lora(remapped_path, lora_sd)
                                    logger.info(f"'{name}': saved extended-remap copy to {remapped_path}")
                                except Exception as e:
                                    logger.warning(f"'{name}': failed to save extended-remap copy: {e}")
                    else:
                        logger.info(
                            f"'{name}': no manifest available for {lora_block_count}->{model_block_count} "
                            f"blocks, applied as-is"
                        )
                        remap_info_lines.append(
                            f"{name}: no manifest for {lora_block_count}->{model_block_count}, applied as-is"
                        )
                else:
                    logger.info(f"'{name}': applied as-is (remap not needed/enabled)")
                    remap_info_lines.append(
                        f"{name}: applied as-is (blocks={lora_block_count}, model={model_block_count})"
                    )

            # A LoRA that references more blocks than the connected model has
            # cannot be remapped DOWN -- there's nowhere for its high-index
            # tensors to go. Rather than silently letting
            # comfy.sd.load_lora_for_models skip those tensors, stop here.
            if model_block_count is not None:
                lora_block_count_final = get_lora_block_count(lora_sd.keys())
                if lora_block_count_final is not None and lora_block_count_final > model_block_count:
                    raise AnimaBlockMismatchError(
                        f"'{name}' references {lora_block_count_final} blocks but the "
                        f"connected model only has {model_block_count}. It cannot be remapped "
                        f"down onto a smaller model, so this run is stopped rather than "
                        f"silently applying a partial/broken LoRA."
                    )

            w_model_final = w_model * weight_multiplier
            w_clip_final = w_clip * weight_multiplier

            out_model, out_clip = comfy.sd.load_lora_for_models(
                out_model, out_clip, lora_sd, w_model_final, w_clip_final
            )

        return (out_model, out_clip, stripped_text, "\n".join(remap_info_lines))


NODE_CLASS_MAPPINGS = {
    "AnimaLoRARemapExtendedTagLoader": AnimaLoRARemapExtendedTagLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoRARemapExtendedTagLoader": "Anima LoRA Tag Loader Extended (Experimental)",
}
