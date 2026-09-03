# ComfyUI-Anima-Remap

[English](README.md) | [日本語](README_ja.md)

> **Note (Anima-3.8B v1.1+):** Some Anima checkpoints ship the Qwen3.5 4B cross-attention component (the `semantic_attentions.*` / connector keys) bundled inside the same file as the DiT, rather than as a separate adapter file — this changed with Anima-3.8B v1.1's "Semantic Connector v2". Regardless of which way it's packaged, this component is never touched by remapping: detection and remapping only ever look at `net.blocks.N` keys, and the connector's keys don't match that pattern. Confirmed working: Qwen3.5 can be left disconnected entirely and generation still works via the native Qwen3 0.6B path. See "About Anima-3.8B (52 blocks) support" below.

A ComfyUI custom node package for applying LoRAs and merging models across the Anima model family (original Anima/28 blocks, Anima-2.9B/40 blocks, Anima-3.8B/52 blocks — and whatever comes next), automatically accounting for the difference in block/layer structure between whichever two generations you're working with. Also includes Anima-specific random LoRA loaders (folder-based, with the same auto-remap logic built in).

This is the successor to [ComfyUI-Anima29B-Remap](https://github.com/shin131002/ComfyUI-Anima29B-Remap), which is no longer updated. Node IDs are unchanged, so existing workflows built against that repository keep loading here without modification.

> ⚠️ **Do not install both this repository and the old `ComfyUI-Anima29B-Remap` at the same time.** They register the same node IDs, so having both installed will cause a node-registration conflict. If you're migrating, remove the old one first.

## What this is for

Each new Anima generation so far has been created by inserting new blocks in between the blocks of the previous generation (Anima's 28 → Anima-2.9B's 40 → Anima-3.8B's 52). Because of this, applying a LoRA or merging a model made for an earlier generation directly onto a later one causes the **block indices to line up incorrectly, so weights get applied to the wrong blocks entirely** — resulting in broken, "scribble-level" images, or noisy output from model merges.

This package uses `expand_manifest.json` files (which record how block indices map during each expansion) to automatically detect and correct this mismatch before applying a LoRA or merging models — regardless of which two generations you're bridging.

## Folder structure

```
ComfyUI-Anima-Remap/
├── LICENSE                                  # Code license (MIT)
├── README.md                                # This file (English)
├── README_ja.md                             # Japanese README
├── __init__.py                              # Node registration
├── mapping/
│   ├── expand_manifest_28_40.json           # Official Anima -> Anima-2.9B manifest
│   ├── expand_manifest_40_52.json           # Anima-2.9B -> Anima-3.8B (52 blocks) manifest (reconstructed, see provenance note below)
│   ├── expand_manifest_28_52_composed.json  # Anima -> Anima-3.8B, auto-composed from the two above
│   └── scripts/
│       └── compose_manifests.py             # Maintenance tool: regenerates every pairwise manifest from whatever adjacent-generation manifests exist. Not used at runtime by any node.
└── nodes/
    ├── __init__.py                          # (empty, makes this a proper package)
    ├── anima_common.py                      # Shared logic (block detection, manifest auto-selection, mapping computation)
    ├── lora_remap_anima.py                  # LoRA tag loader (auto remap)
    ├── lora_remap_extended_anima.py         # LoRA tag loader, extended (experimental, front/back blend)
    ├── model_merge_anima.py                 # Model merge (auto remap)
    ├── model_merge_extended_anima.py        # Model merge, extended (experimental, front/back blend)
    ├── anima_random_lora_loader.py          # Random LoRA loader, 3 folders (auto remap)
    └── anima_filtered_random_lora_loader.py # Random LoRA loader, 1 folder + keyword filter (auto remap)
```

## Installation

Place this entire folder under `ComfyUI/custom_nodes/` and restart ComfyUI.

```
ComfyUI/custom_nodes/ComfyUI-Anima-Remap/
```

Or clone it directly:

```
cd ComfyUI/custom_nodes/
git clone https://github.com/shin131002/ComfyUI-Anima-Remap.git
```

> If you have the old `ComfyUI-Anima29B-Remap` installed, remove that folder first — both packages register the same node IDs, and having both installed at once will cause a node-registration conflict.

After restarting, search for "Anima" in the node search (double-click the canvas) and you'll find these nodes:

- **Anima LoRA Tag Loader (Auto Remap)**
- **Anima Model Merge (Auto Remap)**
- **Anima LoRA Tag Loader Extended (Experimental)** — experimental, see below
- **Anima Model Merge Extended (Experimental)** — experimental, see below
- **Anima Random LoRA Loader (28/40/52 Auto)** — folder-based random selection, 3 groups, see below
- **Anima Filtered Random LoRA Loader (28/40/52 Auto)** — folder-based random selection, 1 folder + keyword filter, see below

---

## Node 1: Anima LoRA Tag Loader (Auto Remap)

A LoRA loader with the same `<lora:name:weight>` tag syntax used by LoRA Tag Power Loader-style nodes, parsed out of a prompt string. For each LoRA tag, it detects both that LoRA's own block count and the connected model's block count, and — if they differ — automatically remaps the LoRA's keys onto the model's generation before applying it.

![Node 1: Anima LoRA Tag Loader (Auto Remap)](./images/01.jpg)

### Inputs

| Name | Type | Description |
|---|---|---|
| `model` | MODEL | The model to apply the LoRA(s) to |
| `clip` | CLIP (optional) | The CLIP to apply the LoRA(s) to |
| `text` | STRING | Prompt string containing `<lora:name:weight>` tags |
| `default_weight` | FLOAT | Default weight used when a tag omits it |
| `weight_multiplier` | FLOAT | A multiplier applied uniformly to every LoRA's weight |
| `auto_remap` | BOOLEAN | When ON, enables the auto-remap mechanism (default: ON) |
| `save_remapped` | BOOLEAN | When ON, **saves the remapped LoRA to disk as a file** whenever a fresh remap is performed (default: OFF — see details and a caution below) |
| `extend_to_new_layers` | BOOLEAN | **Experimental.** When ON, also applies an approximation of the LoRA's effect onto the 12 newly-inserted layers (default: OFF) |
| `extend_strength` | FLOAT | Strength used for `extend_to_new_layers` (multiplies together with the tag's own weight) |
| `manifest` | dropdown | `Auto (Recommended)` (default) resolves the right manifest per LoRA tag, based on that LoRA's own block count vs. the connected model's. A mixed prompt referencing both a 28-block and a 40-block LoRA against a 52-block model correctly uses a different manifest for each. Selecting a specific file instead forces that one for every tag, regardless of what's detected |

### Outputs

| Name | Type | Description |
|---|---|---|
| `model` | MODEL | Model with the LoRA(s) applied |
| `clip` | CLIP | CLIP with the LoRA(s) applied |
| `text` | STRING | The prompt string with LoRA tags stripped out (feed this to your downstream CLIP Text Encoder) |

### Tag syntax

```
1girl, masterpiece <lora:my_old_anima_style:0.8> outdoors
```

You can also specify the CLIP weight separately with `<lora:name:model_weight:clip_weight>` (if omitted, it defaults to the model weight).

### Auto-detection logic (overview)

For each `<lora:name:weight>` tag, independently:

1. Detect the total block count of the connected `model` (max index + 1 found among `net.blocks.N` keys; `llm_adapter.blocks` keys are excluded from this detection)
2. Detect how many blocks that specific LoRA file has keys for, the same way
3. If the LoRA has fewer blocks than the model, resolve the manifest for that exact (LoRA blocks -> model blocks) pair (see `manifest` above) and remap. If the LoRA already matches or exceeds the model's block count, it's applied as-is (see the mismatch note below for the "exceeds" case)

Because this happens per tag, a single prompt referencing LoRAs from different Anima generations against the same model is handled correctly — each gets remapped (or not) according to its own detected generation, not a single decision made once for the whole node run.

> ⚠️ **If a LoRA references *more* blocks than the connected model has** (e.g. a LoRA made for Anima-3.8B applied to an Anima-2.9B model), remapping it down isn't possible — there's nowhere for its high-index tensors to go. Rather than silently letting ComfyUI skip those unmatched tensors and produce a partial/broken application, the node **raises an error and stops the run**. This is checked regardless of which code path above produced the final LoRA (cache hit, freshly remapped, or as-is).

### About the remap cache (`save_remapped`)

`save_remapped` is a toggle that controls **whether the remapped LoRA gets saved to disk as a file**. Saving it lets subsequent runs skip the remapping step entirely and just load the saved file directly.

The cache filename encodes the **target block count it was remapped for** (e.g. `<name>_animaremap52.<ext>`), not just the LoRA's name — so a cache made while targeting a 40-block model is never mistakenly reused if you later connect a 52-block model with the same LoRA. Each (LoRA, target generation) pair gets its own cache file.

> ⚠️ **Caution: while you're still tuning `manifest`, `extend_strength`, or `extend_to_new_layers`, we strongly recommend keeping `save_remapped` OFF.**
>
> Once a cache file has been saved even once, **any later changes to `manifest`, `extend_to_new_layers`, or `extend_strength` will have no effect at all** for that (LoRA, target) pair. As long as the cache file exists, its contents — baked in at whatever settings were active the moment it was saved — will keep being loaded and take priority over your current node settings.
>
> The recommended workflow is:
>
> 1. Keep `save_remapped` **OFF** while you experiment with `manifest` / `extend_to_new_layers` / `extend_strength` to find settings you like (during this phase, the LoRA is remapped fresh on every run and nothing is written to disk)
> 2. Once you've settled on values, turn `save_remapped` **ON** for a single run to write out the final cache file
> 3. If you want to try different settings again later, delete the generated cache file first, then go back to step 1

This is the full sequence of events for a given tag (which only occurs when the LoRA is determined to need remapping for the connected model's block count):

1. **First, check whether a file named `<original LoRA name>_animaremap<target block count>.<extension>` already exists**, using the same lookup method as for the original LoRA (including subfolders)
2. **If it exists**: load that file directly and apply it. No remapping is performed at all — an existing cache file always takes priority, regardless of the `save_remapped` setting
3. **If it doesn't exist**: load the original LoRA, remap its keys on the fly, and apply it. If `save_remapped` is ON at this point, the remapped result is saved as `<original LoRA name>_animaremap<target block count>.<extension>` **in the same folder as the original LoRA** (if a file with that name already exists for some other reason, it is not overwritten — the save is skipped). If `save_remapped` is OFF, the LoRA is still applied, but nothing is written to disk (so this "remap on the fly" step will happen again on every subsequent run)

In short, leaving `save_remapped` ON means the cache file generated on the first run keeps getting reused afterward for that same (LoRA, target generation) combination, so **every subsequent run applies the LoRA with no remapping overhead**.

### Notes

- `llm_adapter.blocks` (6 blocks, not part of any expansion so far) is treated as a completely separate structure from the main `net.blocks` and is always excluded from remapping
- LoRA key naming is supported in two forms: dot-separated (`net.blocks.N.`) and kohya-style underscore-separated (`..._blocks_N_...`). Any other naming convention won't be detected, and the LoRA will be applied unmodified without remapping
- Nodes 5 & 6 (the random loaders) scan for `.safetensors`, `.pt`, and `.ckpt` files

---

## Node 2: Anima Model Merge (Auto Remap)

Merges two Anima models (MODEL), automatically reconciling any difference in block/layer structure between them — regardless of which two generations they're from.

![Node 2: Anima Model Merge (Auto Remap)](./images/02.jpg)

### Inputs

| Name | Type | Description |
|---|---|---|
| `model_1` | MODEL | First model to merge (top slot) |
| `model_2` | MODEL | Second model to merge (bottom slot) |
| `merge_ratio` | FLOAT (0.0–1.0) | Blend weight for `model_1` |
| `extend_ratio` | FLOAT (0.0–1.0) | **Experimental.** How much of the corresponding smaller-generation layer to blend into the newly-inserted blocks (default: 0.0 — new layers stay 100% the larger generation's own values, as before) |
| `manifest` | dropdown | `Auto (Recommended)` (default) resolves the manifest from `model_1`/`model_2`'s detected block counts. Selecting a specific file forces that one instead |

### Outputs

| Name | Type |
|---|---|
| `model` | MODEL |

### What `merge_ratio` means

- `1.0` → output is 100% `model_1` (top)
- `0.0` → output is 100% `model_2` (bottom)
- `0.5` → a 50/50 blend

### Automatic architecture handling

| Situation | Output |
|---|---|
| Both models have the same block count (both derivatives of the same generation) | Direct merge at that block count (no remap needed) |
| Block counts differ | **Output is always the larger block count.** The smaller-generation model's weights are remapped and blended in at the shared block positions according to `merge_ratio`. By default, the newly-inserted blocks always keep the larger model's own values — regardless of whether that model is plugged into `model_1` or `model_2` — and are unaffected by `merge_ratio` (this can be changed with `extend_ratio`, see below) |
| Block counts differ but no manifest covers that specific pair | Falls back to an unremapped direct merge, with a warning logged — results may be incorrect if the architectures actually differ |

### About `extend_ratio` (extending to the newly-inserted layers, experimental)

By default (`extend_ratio = 0.0`), the newly-inserted blocks always keep the larger-generation model's own values, unaffected by `merge_ratio` or the smaller-generation model at all.

Setting `extend_ratio` above 0 uses the resolved manifest's `inserted_to_source` (which records which base block each new block was originally copied from at initialization) to blend in the smaller model's corresponding (remapped) source layer:

```
final new-layer value = (1 - extend_ratio) * larger model's own value + extend_ratio * the smaller model's corresponding source layer
```

At `extend_ratio = 1.0`, the new layers are fully replaced by the smaller model's (approximated) values as well. This follows the same idea as the LoRA node's `extend_to_new_layers` / `extend_strength`, but is an independent feature — and, just like there, it's a best-effort approximation with no "correct" answer.

Since this node has no built-in save function (see below), it doesn't have the same "settings silently stop applying once cached" caveat that the LoRA node's `save_remapped` does — saving a merge result always requires explicitly running the `ModelSave` node, so there's no risk of unknowingly reusing a file baked with outdated settings.

### Bypass behavior

When the node is set to "Bypass" mode, ComfyUI's standard behavior takes over: the first matching input (`model_1`, the top slot) is passed straight through to the output. This is standard ComfyUI behavior for a node with two MODEL inputs and one MODEL output — no special handling is implemented in this node for it.

### Saving the result

This node has no built-in save functionality. To save the merge result, connect the `model` output to ComfyUI's built-in **`ModelSave`** node (found under the `advanced/model_merging` category).

```
[Anima Model Merge] --MODEL--> [ModelSave]
```

`ModelSave` only saves the UNet (diffusion model) portion — CLIP and VAE are not included. By default it saves under `ComfyUI/output`, so move the file to somewhere like `models/diffusion_models` if you want to use it for generation.

---

## Nodes 3 & 4: the Extended (Experimental) variants

`Anima LoRA Tag Loader Extended (Experimental)` and `Anima Model Merge Extended (Experimental)` are **separate files and separate nodes**, added without changing the regular nodes (1 & 2) at all. The regular nodes keep working exactly as before.

![Node 3: Anima LoRA Tag Loader Extended (Experimental)](./images/03.jpg)

![Node 4: Anima Model Merge Extended (Experimental)](./images/04.jpg)

### What's extended

The regular nodes' `extend_to_new_layers` / `extend_strength` (LoRA) and `extend_ratio` (model merge) only used the single "preceding old layer" recorded in the resolved manifest's `inserted_to_source` when projecting onto newly-inserted layers.

The Extended variants generalize this into **blending both the preceding ("front") and following ("back") old layers**, at a ratio you choose:

- **LoRA Extended**: adds `blend_ratio` (0.0–1.0). The front layer's weight is `blend_ratio`, the back layer's weight is `1 - blend_ratio`.
- **Model Merge Extended**: adds the same `blend_ratio`, used together with the existing `extend_ratio` (`extend_ratio` still controls the mix between "the larger model's own value" and "the blended smaller-model value"; `blend_ratio` now controls how that blended smaller-model value itself is built from the front vs. back layer).

```
value applied to the new layer = front (preceding) layer's value * blend_ratio + back (following) layer's value * (1 - blend_ratio)
```

**At `blend_ratio = 1.0`, the back layer's contribution is zero, so the result is numerically identical to the regular nodes (1 & 2)** (verified). Lowering `blend_ratio` progressively mixes in the following layer's influence.

> **Edge case:** insertion positions near either end of the block sequence may only have ONE of the two neighbors (there's no "front" for the very first inserted block, no "back" for the very last). For those layers, `blend_ratio` doesn't apply — the sole available neighbor is always used at full strength, rather than being scaled down by `blend_ratio` (which could otherwise zero out that layer's extension entirely if `blend_ratio` favored the missing side).

### About the cache file

The LoRA Extended node's cache file uses a suffix that both marks it as the Extended variant's cache AND encodes the target block count it was made for (e.g. `_animaremap52_ext`), so it never collides with the regular node's cache (`_animaremap52`) or with a cache made for a different target generation. As with the regular node, we recommend keeping `save_remapped` OFF while you're still experimenting with `manifest` / `extend_to_new_layers` / `blend_ratio` / `extend_strength` (the same caveat applies here — a cached file reflects only what was selected the moment it was saved).

Both Extended and regular LoRA nodes also share the same per-tag manifest resolution and stop-on-mismatch behavior described above: a LoRA that references more blocks than the connected model has cannot be remapped down, so the node raises an error and stops rather than silently applying it partially.

### When to use these

These are purely for experimentation. Reach for them if you want finer control over how the effect is projected onto the new layers — if the regular nodes already give you what you want, there's no need to switch.

---

## Nodes 5 & 6: Anima Random LoRA Loader / Anima Filtered Random LoRA Loader (28/40/52 Auto)

Anima-only ports of [shin131002/RandomLoRALoader](https://github.com/shin131002/RandomLoRALoader)'s "Random LoRA Loader" (3-folder simultaneous selection) and "Filtered Random LoRA Loader" (1 folder + keyword filter), with the same auto-remap logic as Nodes 1–4 built directly in. Every randomly-selected LoRA is checked and, if needed, remapped before being applied — no manual tagging required.

![Node 5: Anima Random LoRA Loader (28/40/52 Auto)](./images/05.jpg)

![Node 6: Anima Filtered Random LoRA Loader (28/40/52 Auto)](./images/06.jpg)

**Anima Random LoRA Loader** selects LoRAs from up to 3 folders at once (e.g. style / character / concept), each with its own strength range and count. **Anima Filtered Random LoRA Loader** selects from a single folder with keyword filtering (AND/OR, phrase matching, optional metadata search) — better suited to a large, mixed LoRA collection.

Both share the same trigger-word extraction, preview-image output, and strength-range parsing (`"0.4-0.8"` etc.) as the original RandomLoRALoader package. See that project's README for details on those shared features.

### What's different from the original RandomLoRALoader

- **Anima only.** These are not general SD1.5/SDXL/Flux loaders — see [RandomLoRALoader](https://github.com/shin131002/RandomLoRALoader) for those.
- **No LoRA Block Weight (LBW) support.** Anima's flat transformer-block layout doesn't map onto the SD1.5/SDXL IN/MID/OUT preset convention that LBW relies on, so this is out of scope for now.
- **No `save_remapped` / on-disk remap caching.** These nodes are built around picking a different LoRA per run — a cache file would go stale silently the moment `manifest`/`extend_to_new_layers`/`blend_ratio`/`extend_strength` changed, and would also get re-discovered by the folder scan as a second, metadata-less "duplicate" LoRA candidate. Every run remaps in-memory instead; LoRA files are small, so the extra cost is negligible next to generation time.
- **Folder scans exclude any `_animaremap<N>` / `_animaremap<N>_ext` cache files** written by Nodes 1–4. If you've used those with `save_remapped` ON in the same LoRA folders, those cached copies are automatically skipped rather than being treated as extra candidates.
- **Remap settings are ONE shared set**, not per-folder/per-group — `auto_remap`, `manifest`, `extend_to_new_layers`, `blend_ratio`, `extend_strength` apply uniformly across the node. The **manifest that actually gets used is still resolved per LoRA**, though: a folder mixing 28-block and 40-block LoRAs, run against a 52-block model, correctly remaps each one with its own matching manifest when `manifest` is left on `Auto`.
- **Same stop-on-mismatch behavior as Nodes 1–4**: if a randomly-selected LoRA references more blocks than the connected model has, the node raises an error and stops rather than silently applying a partial LoRA.

### Inputs (Random LoRA Loader — Filtered is the same idea with 1 folder + keyword filter instead of 3 folders)

| Name | Type | Description |
|---|---|---|
| `model` / `clip` | MODEL / CLIP | Base model/CLIP |
| `additional_prompt_positive` / `_negative` | STRING | Extra prompt text, combined with each selected LoRA's trigger words |
| `lora_folder_path_1..3` | STRING | Folder path per group |
| `include_subfolders_1..3` | BOOLEAN | Recurse into subfolders |
| `unique_by_filename_1..3` | BOOLEAN | Exclude duplicate filenames across subfolders |
| `model_strength_1..3` / `clip_strength_1..3` | STRING | Fixed (`"1.0"`) or random range (`"0.4-0.8"`) |
| `num_loras_1..3` | INT | How many LoRAs to pick per group |
| `trigger_word_source` | dropdown | `json_combined` / `json_random` / `json_sample_prompt` / `metadata` |
| `seed` | INT | Random selection seed |
| `auto_remap` | BOOLEAN | Shared across the node (default: ON) |
| `extend_to_new_layers` | BOOLEAN | **Experimental**, shared across the node (default: OFF) |
| `blend_ratio` | FLOAT | Shared across the node (default: 1.0) |
| `extend_strength` | FLOAT | Shared across the node (default: 0.5) |
| `manifest` | dropdown | `Auto (Recommended)` (default) resolves the manifest per selected LoRA. Selecting a specific file forces that one for every LoRA instead |

### Outputs

Same shape as the original RandomLoRALoader nodes: `MODEL`, `CLIP`, `positive_text`, `negative_text`, `positive` (CONDITIONING), `negative` (CONDITIONING), `preview` (IMAGE). `positive_text` includes each selected LoRA as `<lora:name:model_strength:clip_strength>, trigger words` (LoRA syntax is stripped before it reaches the CONDITIONING outputs).

---

## About Anima-3.8B (52 blocks) support

[lylogummy/Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B) (52 blocks) is a community expansion of Anima-2.9B to 52 blocks (40 native + 12 newly-trained), paired with an optional Qwen3.5 4B cross-attention component for improved prompt adherence. Depending on the release, this component ships either as a separate adapter file (early preview builds) or bundled directly into the main DiT checkpoint (Anima-3.8B v1.1's "Semantic Connector v2" and later) — this packaging has changed at least once already and may change again. This package only concerns itself with the **DiT block structure** of the 52-block checkpoint. Either way it's packaged, the Qwen3.5 component is an additional, self-contained set of keys (`semantic_attentions.*` and related connector keys, not matching any `net.blocks.N` pattern) and isn't touched by remapping at all — detection and remapping only ever look at `net.blocks.N` keys. LoRAs and merges made for the native 40-block Anima-2.9B DiT remap onto the 52-block DiT exactly the same way earlier-generation LoRAs remap onto Anima-2.9B. Confirmed working in practice: Qwen3.5 can be left disconnected entirely (native Qwen3 0.6B path only) and generation still works.

> **Provenance note:** unlike the official Anima → Anima-2.9B `expand_manifest.json`, no official block-mapping file has been published for the Anima-2.9B → Anima-3.8B expansion. `mapping/expand_manifest_40_52.json` was reconstructed by comparing `anima38B_base.safetensors` against an Anima-2.9B checkpoint block-by-block (cosine similarity on `self_attn.q_proj.weight` and `mlp.layer1.weight`, cross-validated against each other), which reveals the LLaMA Pro-style "copy an adjacent block, then fine-tune" initialization pattern. If an official mapping is ever published, replace this file with it; `mapping/expand_manifest_28_52_composed.json` should then be regenerated too (see below).

### ⚠️ Text encoder compatibility caveat (Qwen3.5 4B)

This package's remap logic only concerns the **DiT block structure**. It has no visibility into, and doesn't touch, how the connected text encoder path affects generation — that's a separate, independent axis from block-remapping, but it matters enough for Anima-3.8B specifically that it's worth stating plainly here:

- **Whether the Qwen3.5 4B adapter functions at all is a question of generation (block count), and it works identically across all of them.** The adapter injects into the `cross_attn` slot that every block already has as part of the standard Anima block template — a slot that predates Anima-3.8B and is unchanged by block-remapping. So this isn't specific to 52 blocks: it works the same on 28 and 40 blocks too.
- **Whether the results are good is a separate question of the text encoder's own characteristics, unrelated to block count.** Qwen3.5 4B is a substantially more capable language model than Qwen3 0.6B (the encoder Anima/Anima-2.9B LoRAs were originally trained against), with much stronger literal instruction-following. That difference in capability — not which generation's DiT it's attached to — is what determines how well a given LoRA or prompt carries over when you switch text encoders.

None of this is specific to whether a LoRA needed block-remapping in the first place. It's stated here because Anima-3.8B is the first generation in this family to ship with an optional second text encoder, making this distinction relevant for the first time.

## About `expand_manifest.json` and multi-generation support

Each file in `mapping/` records how block indices map during one Anima expansion step (`old_block_count`, `new_block_count`, `insertion_positions`, and `inserted_to_source` — which base block each new block was originally copied from at initialization). Every node's `manifest` dropdown defaults to `Auto (Recommended)`, which picks whichever manifest's `(old_block_count, new_block_count)` matches the LoRA/model pair it's actually working with, so you generally never need to touch this dropdown.

`mapping/expand_manifest_28_52_composed.json` isn't hand-authored — it's mechanically composed from the 28→40 and 40→52 manifests by `mapping/scripts/compose_manifests.py`, so that a 28-block LoRA can be remapped directly onto a 52-block model in one step, with no runtime chaining logic needed in the nodes themselves.

If a new Anima generation appears in the future with its own block expansion:

1. Add that generation's manifest (official if published, reconstructed via the same kind of comparison otherwise) to `mapping/` as `expand_manifest_<old>_<new>.json`
2. Run `python mapping/scripts/compose_manifests.py auto mapping/` to regenerate every newly-possible pairwise combination against what's already there
3. Nothing else changes — every node picks up the new manifest(s) automatically via `Auto`

## Known limitations

- Officially-published block mappings only exist for the original Anima → Anima-2.9B expansion; anything beyond that (currently, → Anima-3.8B/52 blocks) relies on a reconstructed mapping — see the provenance note above
- If a LoRA's key naming doesn't match an expected pattern, auto-detection fails and it won't be remapped
- **Generation detection is based on the highest block index a LoRA's own keys actually reference, not a label baked into the file.** A LoRA trained for a later generation that happens to only touch early/mid blocks (e.g. a structure-only LoRA) could be misidentified as an earlier generation's LoRA and get remapped when it shouldn't be. This is a known edge case with no override option yet beyond manually forcing a specific `manifest` file; if it turns out to matter in practice, a dedicated "force generation X" override could be added later
- Model merging uses `comfy.model_patcher.ModelPatcher`'s `get_key_patches` / `add_patches` — the same mechanism ComfyUI's own built-in merge nodes use

## About licensing

Anima, Anima-2.9B, and Anima-3.8B (52 blocks) are all distributed under the **CircleStone Labs Non-Commercial License** — Anima-3.8B's own model card states it follows the original Anima-base/Anima-2.9B licenses. Any remapped LoRA files or merged model files produced using this package (the LoRA Remap / Model Merge nodes), for any of these generations, are "Derivatives" under that license, and therefore **inherit the same non-commercial restriction**.

- The model itself and its derivatives (including remapped LoRAs and merged models produced here) may only be used for non-commercial purposes
- However, **images generated (Outputs) using these models can be used commercially** — the license explicitly excludes generated Outputs from the definition of "Derivative"
- Anima is also a derivative model of `Cosmos-Predict2-2B-Text2Image`, so the NVIDIA Open Model License Agreement applies as well, to the extent it covers derivative models
- The license does make an exception allowing an individual to sell "the model weights themselves" (e.g. a LoRA or merged model file) (Section 2.c), but this does not extend to a product, service, or tool built around the model — that would require a separate commercial license
- Anima-3.8B's separate Qwen3.5 4B text-encoder component is Apache 2.0 (consistent with Qwen's licensing for models in that size range) and isn't processed or redistributed by this package in any form, so it doesn't add any restriction here — it's mentioned only for completeness

This tool is intended for personal, non-commercial use. If you're considering commercial use or distribution, always check the primary source — `LICENSE.md` in the Anima repository on Hugging Face, and the license notes on each specific model's page — or consult a professional. Nothing in this README constitutes legal advice.

Note that this license restriction applies to **the Anima model weights themselves** (and to any remapped LoRAs or merged models produced with them) — **the code in this repository** (the Python node implementations) is released under the **MIT License** (see the bundled `LICENSE` file).

## Disclaimer and Support Policy

### Disclaimer

- This node is provided **without technical support**
- No guarantee of functionality
- Compatibility with future ComfyUI updates is not guaranteed
- Bug reports and feature requests may not be addressed
- Use at your own risk

### Support Status

- ❌ No individual support via issues or email
- ❌ No guarantee of bug fixes or new features
- ✅ Code is open source - fork and modify freely
- ✅ Community discussion welcome (no guarantee of a response)

### Reporting Issues

Support isn't guaranteed, but you can:
1. Check existing issues in the repository
2. Check this README and its troubleshooting section
3. Open an issue (it may not be addressed)
4. Fork it and fix it yourself

## License

MIT License - free to use, modify, and distribute.

That said, as noted above, the Anima model itself (its weights, and any LoRAs or merged models produced from them) remains subject to the separate CircleStone Labs Non-Commercial License.
