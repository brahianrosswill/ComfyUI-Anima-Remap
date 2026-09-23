"""
Block-index remapping for ControlNet-LLLite checkpoints.

A LLLite checkpoint stores one entry per patched module, named after the module's
position in the DiT it was trained against:

    lllite_dit_blocks_19_self_attn_q_proj.down.weight

The node that applies it builds its module list from the CONNECTED model, so on a
40-block model it looks for `blocks_0` .. `blocks_39` while a checkpoint trained on
the 28-block base only has `blocks_0` .. `blocks_27` -- and block 19 means a
different physical block in each architecture anyway. Renaming the block indices
with the same expand manifests the LoRA nodes use puts every trained module back
on the block it was actually trained for.

Upward (28 -> 40/52): the 12 blocks inserted per expansion step have no trained
counterpart. By default they are left without weights, which is an exact no-op;
optionally (extend_lllite_state_dict) they receive a copy of the module on the
block they were initialised from.
Downward (40 -> 28): the checkpoint's inserted-block entries have nowhere to go on
the smaller model and are dropped.
"""

import logging
import re

from .anima_common import (
    computable,
    build_base_to_target,
    build_insertion_neighbors,
    resolve_manifest,
    resolve_manifest_filename,
)

logger = logging.getLogger("AnimaRemap")

# "lllite_dit_blocks_19_self_attn_q_proj.down.weight" -> prefix / 19 / rest
LLLITE_BLOCK_PATTERN = re.compile(r"^(lllite_dit_blocks_)(\d+)(_.*)$")

KNOWN_BLOCK_COUNTS = (28, 40, 52)


def find_lllite_block_indices(keys):
    out = set()
    for k in keys:
        m = LLLITE_BLOCK_PATTERN.match(k)
        if m:
            out.add(int(m.group(2)))
    return out


def get_lllite_block_count(keys):
    """
    Which architecture the checkpoint was trained against, bucketed to the smallest
    known Anima block count that could contain its highest block index.

    Bucketing rather than "max + 1" matters because a checkpoint need not patch
    every block -- a 28-block checkpoint whose highest patched block is 25 must
    still be treated as 28, not 26.
    """
    indices = find_lllite_block_indices(keys)
    if not indices:
        return None
    highest = max(indices) + 1
    for count in KNOWN_BLOCK_COUNTS:
        if highest <= count:
            return count
    return highest


def get_host_block_count(lllite):
    """
    Block count of the model the LLLite modules were just built on.

    Read from the module list rather than guessed from the model's state dict:
    ControlNetLLLiteDiT creates one module per matching DiT sub-layer by walking
    the connected model, so the highest block index among its module names IS the
    host architecture. This must never come back None when modules exist -- the
    zero-fill in the vendored loader means an undetected mismatch would silently
    put every module on the wrong block, where upstream would have errored.
    """
    indices = find_lllite_block_indices(
        getattr(m, "lllite_name", "") for m in getattr(lllite, "lllite_modules", [])
    )
    return (max(indices) + 1) if indices else None


def build_lllite_block_map(manifest_choice, source_count, target_count):
    """
    {checkpoint_block_idx: host_block_idx}, or (None, None) when nothing to do.

    Returns (mapping, manifest_filename). A checkpoint block absent from the
    mapping has no counterpart on the host and its weights are dropped.
    """
    if source_count is None or target_count is None or source_count == target_count:
        return None, None

    small, large = min(source_count, target_count), max(source_count, target_count)
    manifest = resolve_manifest(manifest_choice, small, large)
    if manifest is None:
        return None, None
    manifest_name = resolve_manifest_filename(manifest_choice, small, large)

    base_to_target = build_base_to_target(manifest)
    if source_count < target_count:
        # Expanding: the checkpoint's own numbering IS the base numbering.
        return dict(base_to_target), manifest_name
    # Reducing: the host is the base, so read the same table backwards. Checkpoint
    # blocks that aren't a value in the table are inserted blocks with no home here.
    return {target_idx: base_idx for base_idx, target_idx in base_to_target.items()}, manifest_name


def remap_lllite_state_dict(weights_sd, block_map):
    """
    Rewrite block indices in the checkpoint's keys. Non-block keys (conditioning1.*)
    pass through untouched. Returns (new_state_dict, renamed_count, dropped_count).
    """
    if not block_map:
        return weights_sd, 0, 0

    out, renamed, dropped = {}, 0, 0
    for k, v in weights_sd.items():
        m = LLLITE_BLOCK_PATTERN.match(k)
        if not m:
            out[k] = v
            continue
        source_idx = int(m.group(2))
        if source_idx not in block_map:
            dropped += 1
            continue
        out[f"{m.group(1)}{block_map[source_idx]}{m.group(3)}"] = v
        renamed += 1
    return out, renamed, dropped


def build_lllite_extension_sources(manifest_choice, source_count, target_count):
    """
    {inserted_host_block: host_block_to_copy_from}, only when expanding.

    The source is the nearest PRECEDING original block. That isn't an arbitrary
    pick: each inserted Anima block was deep-copied from its immediate predecessor
    when the model was expanded (see build_insertion_neighbors), so that
    predecessor's control module is the one trained for the closest function.
    Falls back to the following block only if nothing precedes (never the case
    for the shipped manifests).
    """
    if source_count is None or target_count is None or source_count >= target_count:
        return None
    manifest = resolve_manifest(manifest_choice, source_count, target_count)
    if manifest is None:
        return None
    base_to_target = build_base_to_target(manifest)
    out = {}
    for inserted, (prev_base, next_base) in build_insertion_neighbors(manifest).items():
        base = prev_base if prev_base is not None else next_base
        if base is not None and base in base_to_target:
            out[inserted] = base_to_target[base]
    return out


def extend_lllite_state_dict(weights_sd, extension_sources, strength):
    """
    Give each inserted block a copy of its source block's modules. Run AFTER
    remap_lllite_state_dict, so keys are already in host numbering.

    Every tensor of the module is copied whole -- down, mid, cond_to_film, up and
    depth_embed together. Averaging two neighbours instead would be wrong here:
    a module's correction is up(mid(down(x), ...)), a product of low-rank factors,
    and averaging factors from two different modules does not average their
    corrections -- it mixes two unrelated bases into a direction neither learned.

    `strength` scales only up.weight / up.bias. `up` is the module's final layer
    (out = up(m) * multiplier), so this scales the copied module's contribution
    by exactly `strength` and nothing else.

    Returns (new_state_dict, extended_module_count).
    """
    if not extension_sources:
        return weights_sd, 0

    by_block = {}
    for k, v in weights_sd.items():
        m = LLLITE_BLOCK_PATTERN.match(k)
        if m:
            by_block.setdefault(int(m.group(2)), []).append((m.group(1), m.group(3), v))

    out = dict(weights_sd)
    extended_modules = set()
    for inserted, source in extension_sources.items():
        for prefix, rest, v in by_block.get(source, []):
            new_key = f"{prefix}{inserted}{rest}"
            if new_key in out:
                continue  # never overwrite real trained weights
            if rest.endswith(".up.weight") or rest.endswith(".up.bias"):
                v = computable(v) * strength
            out[new_key] = v
            extended_modules.add(new_key.split(".", 1)[0])
    return out, len(extended_modules)
