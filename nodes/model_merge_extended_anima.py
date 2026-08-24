"""
model_merge_extended_anima.py

EXPERIMENTAL variant of AnimaModelMerge. Same base merge behavior for the
shared (old) 28 blocks, but generalizes how the 28-block model's weights
are projected onto Anima-2.9B's 12 newly-inserted layers: instead of always
using the single "source" layer recorded in expand_manifest.json (always
the immediately-PRECEDING old layer), this node blends between the
preceding ("front") and following ("back") old layers using `blend_ratio`
before applying extend_ratio.

    old_model_blended  = front_layer * blend_ratio + back_layer * (1 - blend_ratio)
    new_layer (final)  = (1 - extend_ratio) * 2.9B's own value + extend_ratio * old_model_blended

At blend_ratio = 1.0, this is mathematically identical to the original
node's extend_ratio behavior (100% front, 0% back). Lower values
progressively mix in the following layer instead.

This is a SEPARATE node (not a modification of model_merge_anima.py) so
the original, simpler node keeps working exactly as before.
"""

import logging

from .anima_common import (
    remap_key,
    split_block_key,
    list_manifest_choices,
    resolve_manifest,
    build_base_to_target,
    build_insertion_neighbors,
    get_model_block_count,
)
from .model_merge_anima import remap_key_patches

logger = logging.getLogger("AnimaModelMergeExtended")


def group_patches_by_base_index(key_patches):
    """{base_idx: {suffix: (prefix, sep, patch)}} for every MAIN-block patch."""
    groups = {}
    for k, v in key_patches.items():
        parsed = split_block_key(k)
        if parsed is None:
            continue
        prefix, base_idx, suffix, sep = parsed
        groups.setdefault(base_idx, {})[suffix] = (prefix, sep, v)
    return groups


def build_front_back_extension_patches(groups, neighbors):
    """
    For each inserted target layer, collect its front (preceding) and back
    (following) neighbor patches SEPARATELY, re-keyed onto the new target
    index, WITHOUT any arithmetic on the patch values themselves (patches
    from get_key_patches() are ComfyUI's internal patch format, not raw
    tensors, so they can't be scalar-multiplied directly -- blending has to
    happen via add_patches()'s own strength_patch/strength_model instead).

    Near either end of the block sequence, an inserted layer may have only
    ONE of the two neighbors (see build_insertion_neighbors). Those layers
    are split out into solo_front/solo_back instead of front_patches/
    back_patches, because they must be applied at full strength (there is no
    "other side" for blend_ratio to weigh against) -- lumping them into the
    same dicts as the two-sided case would let blend_ratio silently zero
    them out when it favors the missing side.

    Returns (front_patches, back_patches, solo_front, solo_back), each
    {new_key: patch}. front_patches/back_patches may share key names (a
    t_idx with both neighbors contributes to both); solo_front/solo_back
    never overlap with each other or with front_patches/back_patches.
    """
    front_patches = {}
    back_patches = {}
    solo_front = {}
    solo_back = {}
    for t_idx, (prev_base, next_base) in neighbors.items():
        has_prev = prev_base is not None and prev_base in groups
        has_next = next_base is not None and next_base in groups
        if has_prev and has_next:
            for suffix, (prefix, sep, patch) in groups[prev_base].items():
                front_patches[f"{prefix}{t_idx}{sep}{suffix}"] = patch
            for suffix, (prefix, sep, patch) in groups[next_base].items():
                back_patches[f"{prefix}{t_idx}{sep}{suffix}"] = patch
        elif has_prev:
            for suffix, (prefix, sep, patch) in groups[prev_base].items():
                solo_front[f"{prefix}{t_idx}{sep}{suffix}"] = patch
        elif has_next:
            for suffix, (prefix, sep, patch) in groups[next_base].items():
                solo_back[f"{prefix}{t_idx}{sep}{suffix}"] = patch
    return front_patches, back_patches, solo_front, solo_back


def apply_blended_extension(m, groups, neighbors, blend_ratio, extend_ratio):
    """
    Apply the front/back-blended extension onto `m` in place, using
    sequential add_patches() calls so the actual blending math is done by
    ComfyUI itself (never by us touching patch values directly):

        two-sided layers -- 1st call (front): new = (1 - extend_ratio) * current + (extend_ratio * blend_ratio) * front
                             2nd call (back):  new = 1.0 * current + (extend_ratio * (1 - blend_ratio)) * back
        one-sided layers  -- single call: new = (1 - extend_ratio) * current + extend_ratio * (the only side)
                             (blend_ratio does not apply -- there is nothing to weigh it against)

    Returns the total number of tensors touched.
    """
    front_patches, back_patches, solo_front, solo_back = build_front_back_extension_patches(groups, neighbors)

    # front_patches/back_patches share key names for two-sided layers, so the
    # (1 - extend_ratio) shrink of the model's current value must only be
    # applied on the first of the two calls that actually touches those keys.
    shrink_applied = False
    if front_patches:
        m.add_patches(front_patches, extend_ratio * blend_ratio, 1.0 - extend_ratio)
        shrink_applied = True
    if back_patches:
        shrink = 1.0 if shrink_applied else (1.0 - extend_ratio)
        m.add_patches(back_patches, extend_ratio * (1.0 - blend_ratio), shrink)

    # solo_front/solo_back are disjoint from front_patches/back_patches and
    # from each other, so each gets its own independent, single, full-strength
    # touch (relative to extend_ratio) with its own (1 - extend_ratio) shrink.
    if solo_front:
        m.add_patches(solo_front, extend_ratio, 1.0 - extend_ratio)
    if solo_back:
        m.add_patches(solo_back, extend_ratio, 1.0 - extend_ratio)

    return len(front_patches) + len(back_patches) + len(solo_front) + len(solo_back)


class AnimaModelMergeExtended:
    """
    EXPERIMENTAL: same as AnimaModelMerge, but the extension onto
    Anima-2.9B's 12 newly-inserted layers uses a continuous front/back
    blend_ratio (applied to the 28-block model's neighboring layers)
    instead of always using the single preceding layer. blend_ratio=1.0
    reproduces the original node's extend_ratio behavior exactly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        manifests = list_manifest_choices()
        return {
            "required": {
                "model_1": ("MODEL",),
                "model_2": ("MODEL",),
                "merge_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "extend_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "manifest": (manifests,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "merge"
    CATEGORY = "loaders/anima/experimental"

    def merge(self, model_1, model_2, merge_ratio, extend_ratio, blend_ratio, manifest):
        block_count_1 = get_model_block_count(model_1)
        block_count_2 = get_model_block_count(model_2)
        logger.info(f"model_1: {block_count_1} blocks, model_2: {block_count_2} blocks, merge_ratio={merge_ratio}")

        # --- Case 1: same architecture on both sides -> direct merge, no remap ---
        if block_count_1 is not None and block_count_1 == block_count_2:
            m = model_1.clone()
            kp2 = model_2.get_key_patches("diffusion_model.")
            m.add_patches(kp2, 1.0 - merge_ratio, merge_ratio)
            logger.info(f"Direct {block_count_1}-block merge (no remap needed). Output: {block_count_1} blocks.")
            return (m,)

        # --- Case 2: mismatched architectures -> resolve the manifest for
        # THIS SPECIFIC (smaller, larger) pair. The larger side is always
        # the output base; its newly-inserted blocks default to untouched. ---
        if block_count_1 is not None and block_count_2 is not None:
            smaller = min(block_count_1, block_count_2)
            larger = max(block_count_1, block_count_2)
            manifest_data = resolve_manifest(manifest, smaller, larger)
            base_to_target = build_base_to_target(manifest_data) if manifest_data else {}
            neighbors = build_insertion_neighbors(manifest_data) if manifest_data else {}

            if base_to_target:
                if block_count_1 == larger:
                    m = model_1.clone()
                    kp2 = model_2.get_key_patches("diffusion_model.")
                    remapped_kp2, dropped = remap_key_patches(kp2, base_to_target)
                    m.add_patches(remapped_kp2, 1.0 - merge_ratio, merge_ratio)
                    logger.info(
                        f"model_1={larger}(base), model_2={smaller}(old): remapped {len(remapped_kp2)} keys "
                        f"({dropped} dropped), blended at old-block positions, ratio={merge_ratio}."
                    )
                    if extend_ratio > 0.0 and neighbors:
                        groups = group_patches_by_base_index(kp2)
                        n_touched = apply_blended_extension(m, groups, neighbors, blend_ratio, extend_ratio)
                        logger.info(
                            f"[experimental] extended {n_touched} tensors from model_2 (old) onto "
                            f"newly-inserted layers via front/back blend "
                            f"(blend_ratio={blend_ratio}, extend_ratio={extend_ratio})"
                        )
                    return (m,)
                else:
                    m = model_2.clone()
                    kp1 = model_1.get_key_patches("diffusion_model.")
                    remapped_kp1, dropped = remap_key_patches(kp1, base_to_target)
                    m.add_patches(remapped_kp1, merge_ratio, 1.0 - merge_ratio)
                    logger.info(
                        f"model_1={smaller}(old), model_2={larger}(base): remapped {len(remapped_kp1)} keys "
                        f"({dropped} dropped), blended at old-block positions, ratio={merge_ratio}."
                    )
                    if extend_ratio > 0.0 and neighbors:
                        groups = group_patches_by_base_index(kp1)
                        n_touched = apply_blended_extension(m, groups, neighbors, blend_ratio, extend_ratio)
                        logger.info(
                            f"[experimental] extended {n_touched} tensors from model_1 (old) onto "
                            f"newly-inserted layers via front/back blend "
                            f"(blend_ratio={blend_ratio}, extend_ratio={extend_ratio})"
                        )
                    return (m,)

        # --- Fallback: unrecognized/unmapped block counts -> best-effort direct merge ---
        logger.warning(
            f"No manifest covers (model_1={block_count_1}, model_2={block_count_2}) blocks. "
            f"Falling back to an unremapped direct merge -- results may be incorrect if the "
            f"architectures actually differ."
        )
        m = model_1.clone()
        kp2 = model_2.get_key_patches("diffusion_model.")
        m.add_patches(kp2, 1.0 - merge_ratio, merge_ratio)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AnimaModelMergeExtended": AnimaModelMergeExtended,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaModelMergeExtended": "Anima Model Merge Extended (Experimental)",
}
