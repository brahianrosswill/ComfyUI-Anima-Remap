"""
lora_remap_anima.py

A LoRA Tag-style loader (`<lora:name:weight>` syntax embedded in a text
string, same convention as Power Lora Loader / LoRA Tag Power Loader) for
Anima models, with AUTOMATIC block-index remapping when an old-Anima
(28-block) LoRA is applied to Anima-2.9B (40-block, LLaMA Pro-style block
expansion).

How the auto-remap decision is made:
    1. Detect how many transformer blocks the connected MODEL has, by
       scanning its state_dict for ".blocks.<N>." keys (same technique
       ComfyUI core itself uses for Cosmos-Predict2/Anima block-count
       detection -- see Comfy-Org/ComfyUI PR #15555).
    2. Detect how many blocks the LoRA file itself was trained against,
       by scanning its own keys the same way.
    3. If the model has MORE blocks than the LoRA was trained for, the
       LoRA is remapped using the official expand_manifest.json
       (base -> target block index), derived as follows:
         - expand_manifest.json lists which TARGET block indices are the
           newly-inserted ones ("insertion_positions").
         - Every other target index, taken in ascending order, is one of
           the original (frozen, unchanged) base blocks, in order.
         - So base index i maps to the i-th target index NOT in
           insertion_positions.
    4. If the model has the same (or fewer) blocks as the LoRA, or no
       official manifest is found, the LoRA is applied unmodified.

If your LoRA's key naming doesn't match either of the two patterns below
(dot style `net.blocks.N.` or kohya-style `..._blocks_N_...`), it won't be
detected -- extend BLOCK_PATTERNS in that case.
"""

import logging
import os
import re

import folder_paths
import comfy.utils
import comfy.sd
from safetensors.torch import save_file as st_save_file

from .anima_common import (
    get_lora_block_count,
    remap_key,
    remap_key_to_target,
    list_manifest_choices,
    resolve_manifest,
    build_base_to_target,
    build_source_to_inserted_targets,
    get_model_block_count,
    AnimaBlockMismatchError,
    compute_remap_settings_hash,
    peek_lora_block_count,
    resolve_manifest_filename,
)

logger = logging.getLogger("AnimaLoRARemap")

REMAP_SUFFIX = "_animaremap"


def remap_cache_suffix(target_block_count, settings_hash=None):
    """
    The cache filename encodes WHICH target block count a cached remap was made
    for -- e.g. "_animaremap40" vs "_animaremap52" -- so swapping from a
    40-block model to a 52-block model can't silently reuse a cache remapped
    for the wrong target.

    `settings_hash` additionally encodes the manifest and extension settings the
    cache was produced under (see compute_remap_settings_hash). Without it, a
    cache made with one manifest/extend setting would keep being reused after
    those settings changed, with no indication that the new settings were being
    ignored. Omitting it yields the LEGACY suffix, which is only used to detect
    and warn about pre-hash cache files.
    """
    if settings_hash:
        return f"{REMAP_SUFFIX}{target_block_count}_{settings_hash}"
    return f"{REMAP_SUFFIX}{target_block_count}"


def get_remapped_sibling_path(original_path, target_block_count, settings_hash=None):
    """Path for the cached remapped copy: same folder, same extension, target/settings-specific suffix appended before it."""
    base, ext = os.path.splitext(original_path)
    return f"{base}{remap_cache_suffix(target_block_count, settings_hash)}{ext}"


def warn_if_legacy_cache(name, target_block_count, suffix_fn=remap_cache_suffix):
    """
    Pre-hash cache files ("mylora_animaremap52") are deliberately NOT loaded:
    there's no way to tell which manifest/extend settings produced them, which
    is exactly the silent-staleness problem the hash was added to fix. Warn once
    so the user knows the file is now dead weight.
    """
    legacy = resolve_lora_path(f"{name}{suffix_fn(target_block_count)}")
    if legacy is not None:
        logger.warning(
            f"'{name}': ignoring legacy remap cache {os.path.basename(legacy)} "
            f"(pre-hash format -- its manifest/extend settings are unknown). "
            f"Delete it; a new cache will be written under the hashed name."
        )


def save_remapped_lora(path, tensors):
    """Write a remapped LoRA state dict to disk as safetensors. Clones tensors to
    avoid safetensors' 'shared memory' errors on views/slices from the source file."""
    safe_tensors = {k: v.clone().contiguous() for k, v in tensors.items()}
    st_save_file(safe_tensors, path)


# ---------------------------------------------------------------------------
# `<lora:name:weight[:clip_weight]>` tag parsing
# ---------------------------------------------------------------------------

TAG_PATTERN = re.compile(r"<lora:([^:>]+):(-?[\d.]+)(?::(-?[\d.]+))?>")

LORA_EXTENSIONS = [".safetensors", ".pt", ".ckpt", ".sft"]


def _strip_lora_ext(name):
    """
    Remove a trailing LoRA extension, and ONLY a real one.

    os.path.splitext() must not be used here: it strips everything after the
    last dot, whatever that is. Two ways that bites us --
      - a cache lookup name like "mylora.safetensors_animaremap40_a7f6b0"
        would have ".safetensors_animaremap40_a7f6b0" stripped, leaving
        "mylora", which then matches the ORIGINAL LoRA. The cache lookup
        "succeeds" against the unremapped file and no remap is ever applied.
      - a LoRA legitimately named "anima-rl-v0.1" would lose its ".1".
    """
    low = name.lower()
    for ext in LORA_EXTENSIONS:
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def _normalize_lookup_name(name):
    """
    Windows-style separators -> POSIX, plus surrounding whitespace/slashes
    stripped. ComfyUI's own LoRA dropdown shows relative paths using os.sep, so
    on Windows a name copied straight out of it looks like
    `style\\mylora.safetensors` -- which must resolve the same as a hand-typed
    `style/mylora`.
    """
    return (name or "").replace("\\", "/").strip().strip("/")


def resolve_lora_path(name):
    """
    Resolve a LoRA tag name to a full path, tolerating:
      - the extension being omitted (folder_paths.get_full_path needs it exact)
      - the extension being INCLUDED for a file that lives in a subfolder
      - the file living in a subfolder of loras/ (matched by basename)
      - a subfolder path being included, in any case, with either separator
      - case differences anywhere in the name
    Returns None if nothing matches.
    """
    name = _normalize_lookup_name(name)
    if not name:
        return None

    # 1. exact relative-path match (handles names that already include a subfolder/extension)
    path = folder_paths.get_full_path("loras", name)
    if path:
        return path

    # 2. try appending common extensions
    for ext in LORA_EXTENSIONS:
        path = folder_paths.get_full_path("loras", name + ext)
        if path:
            return path

    # 3. fall back to a search across every known lora file.
    #    BOTH sides get normalised the same way before comparing. Previously only
    #    the CANDIDATE had its folder and extension stripped while the user's
    #    input was compared raw, so a name typed WITH its extension
    #    ("mylora.safetensors" for a file inside a subfolder) or with a
    #    differently-cased folder ("Style/mylora") could never match -- steps 1
    #    and 2 don't cover those either, so the lookup failed outright.
    try:
        all_loras = folder_paths.get_filename_list("loras")
    except Exception:
        all_loras = []

    target_rel = _strip_lora_ext(name).lower()
    target_base = os.path.basename(target_rel)

    base_matches = []
    for rel_path in all_loras:
        rel_stem = _strip_lora_ext(_normalize_lookup_name(rel_path)).lower()
        # a full relative path match is unambiguous -- prefer it outright
        if rel_stem == target_rel:
            return folder_paths.get_full_path("loras", rel_path)
        if os.path.basename(rel_stem) == target_base:
            base_matches.append(rel_path)

    if base_matches:
        if len(base_matches) > 1:
            shown = ", ".join(base_matches[:3])
            more = ", ..." if len(base_matches) > 3 else ""
            logger.warning(
                f"'{name}': {len(base_matches)} LoRAs share this basename across subfolders "
                f"({shown}{more}); using the first. Include the subfolder in the tag to "
                f"pick a specific one."
            )
        return folder_paths.get_full_path("loras", base_matches[0])

    return None


def lora_lookup_diagnostic(name):
    """
    Extra context appended to a "LoRA not found" warning.

    Without it, the bare message reads as "the node can't see my loras folder",
    which sends people chasing path/symlink problems when the real cause is
    almost always a name that doesn't match. These two cases need completely
    different fixes, so the warning should say which one it is.
    """
    try:
        all_loras = folder_paths.get_filename_list("loras")
    except Exception as e:
        return f"could not list the loras folder at all ({e})"

    if not all_loras:
        try:
            roots = folder_paths.get_folder_paths("loras")
        except Exception:
            roots = []
        return (
            "ComfyUI reports 0 LoRA files, so this is a folder/config problem rather than a "
            f"name problem -- configured loras path(s): {roots or 'none'}"
        )

    sample = ", ".join(all_loras[:3])
    return (
        f"ComfyUI can see {len(all_loras)} LoRA file(s), so the folder is fine and it's the NAME "
        f"that didn't match. Use the same relative path ComfyUI's own LoRA dropdown shows, "
        f"e.g. {sample}"
    )


def parse_lora_tags(text, default_weight):
    tags = []
    for m in TAG_PATTERN.finditer(text or ""):
        name = m.group(1).strip()
        w_model = float(m.group(2)) if m.group(2) else default_weight
        w_clip = float(m.group(3)) if m.group(3) else w_model
        tags.append((name, w_model, w_clip))
    stripped = TAG_PATTERN.sub("", text or "")
    stripped = re.sub(r"[ \t]{2,}", " ", stripped).strip()
    return tags, stripped


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AnimaLoRARemapTagLoader:
    """
    LoRA Tag-style loader for Anima. Parses `<lora:name:weight>` tags out of
    a text input (same syntax as Power Lora Loader / LoRA Tag Power Loader),
    and -- if the connected MODEL has more transformer blocks than the LoRA
    was trained for -- automatically remaps the LoRA's block indices using
    the bundled expand_manifest.json before applying it. Old-Anima LoRAs
    used on old-Anima models, or LoRAs already trained on Anima-2.9B, pass
    through unchanged.

    Experimental: when extend_to_new_layers is enabled, the LoRA's effect is
    also projected onto Anima-2.9B's newly-inserted layers, by copying each
    inserted layer's nearest-neighbor source layer's (already-remapped) delta
    onto it, scaled by extend_strength. There is no "correct" answer for what
    a pre-2.9B LoRA should do on layers that didn't exist when it was
    trained -- this is a best-effort approximation, off by default.
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
    CATEGORY = "loaders/anima"

    def load(self, model, text, default_weight, weight_multiplier, auto_remap, save_remapped,
              extend_to_new_layers, extend_strength, manifest, clip=None):
        tags, stripped_text = parse_lora_tags(text, default_weight)

        out_model = model
        out_clip = clip
        remap_info_lines = []

        # Same for every tag this run; the MANIFEST itself is resolved per
        # tag below, since different LoRAs can be from different Anima
        # generations (a 28-block LoRA and a 40-block LoRA in the same
        # prompt, both applied to e.g. a 52-block model, need different
        # manifests).
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
            # block count -- read from the safetensors header only, so working
            # out which cache file to look for doesn't cost a full file load
            # (which would defeat the point of the cache).
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
                    )
                    # Built from the RESOLVED path, never from the raw tag text:
                    # the tag may carry a subfolder and/or an extension, and
                    # gluing a suffix onto that produces a name no lookup should
                    # ever have to un-pick.
                    cached_path = resolve_lora_path(
                        f"{cache_stem}{remap_cache_suffix(model_block_count, settings_hash)}"
                    )
                if cached_path is None:
                    warn_if_legacy_cache(cache_stem, model_block_count)

            if cached_path is not None:
                lora_sd = comfy.utils.load_torch_file(cached_path, safe_load=True)
                logger.info(f"'{name}': using cached remap file {os.path.basename(cached_path)}")
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
                    source_to_inserted = build_source_to_inserted_targets(manifest_data) if manifest_data else {}

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

                        if extend_to_new_layers and source_to_inserted:
                            extended = 0
                            for k, v in lora_sd.items():
                                _, base_idx = remap_key(k, base_to_target)
                                if base_idx is None:
                                    continue
                                target_list = source_to_inserted.get(base_idx)
                                if not target_list:
                                    continue
                                for t_idx in target_list:
                                    new_k = remap_key_to_target(k, t_idx)
                                    if new_k is None:
                                        continue
                                    remapped[new_k] = v * extend_strength
                                    extended += 1
                            logger.info(
                                f"'{name}': [experimental] extended {extended} tensors onto "
                                f"newly-inserted layers via nearest-neighbor copy "
                                f"(strength={extend_strength})"
                            )
                            remap_info_lines.append(
                                f"{name}: extended {extended} tensors onto new layers "
                                f"(strength={extend_strength})"
                            )

                        lora_sd = remapped

                        if save_remapped:
                            remapped_path = get_remapped_sibling_path(
                                original_path, model_block_count,
                                settings_hash or compute_remap_settings_hash(
                                    manifest_used, model_block_count,
                                    extend_to_new_layers, extend_strength,
                                ),
                            )
                            if os.path.exists(remapped_path):
                                logger.info(
                                    f"'{name}': remap cache already exists, skipping save: "
                                    f"{os.path.basename(remapped_path)}"
                                )
                            else:
                                try:
                                    save_remapped_lora(remapped_path, lora_sd)
                                    logger.info(
                                        f"'{name}': saved remapped copy to {os.path.basename(remapped_path)}"
                                    )
                                except Exception as e:
                                    logger.warning(f"'{name}': failed to save remapped copy: {e}")
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
    "AnimaLoRARemapTagLoader": AnimaLoRARemapTagLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoRARemapTagLoader": "Anima LoRA Tag Loader (Auto Remap)",
}
