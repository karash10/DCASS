"""
embeddings_legacy/add_nocaps_to_index.py

Adds NoCaps dataset to your existing DCASS image.index — no images
downloaded to disk, no model re-training, no index rebuild.

What this does:
  1. Loads your existing image.index (31,758 vectors) + image_metadata.json
  2. Streams NoCaps from HuggingFace split by split
  3. Embeds each image with the same CLIPEmbedder your project uses
  4. Appends vectors to the FAISS index  (index.add is in-place)
  5. Appends metadata entries that match UnifiedSemanticIndex's schema
  6. Saves the merged index + metadata (originals are backed up first)

Run:
    cd <project_root>
    python -m src.embeddings_legacy.add_nocaps_to_index

Or directly:
    python add_nocaps_to_index.py  (if run from embeddings_legacy/)

Config block at the top — change paths / splits / batch size there.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

# ── resolve project root so relative imports work regardless of CWD ──────────
_HERE        = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent          # src/
_DCASS_ROOT   = _PROJECT_ROOT.parent  # project root (contains storage/)
sys.path.insert(0, str(_DCASS_ROOT))

# ─── CONFIG — edit these to match your layout ─────────────────────────────────
#
# Where unified_index.py looks for indices:
#   project_root/storage/data/indices/   OR
#   project_root/storage/indices/
#
INDICES_DIR   = _DCASS_ROOT / "storage" / "data" / "indices"
if not INDICES_DIR.exists():
    INDICES_DIR = _DCASS_ROOT / "storage" / "indices"

INDEX_FILE    = INDICES_DIR / "image.index"
META_FILE     = INDICES_DIR / "image_metadata.json"

# NoCaps splits to add.  "validation"=~4500 imgs, "test"=~10600 imgs
NOCAPS_SPLITS = ["validation", "test"]

# Batch size for CLIP embedding (lower if you hit OOM)
BATCH_SIZE    = 32

# Set to an int to limit total NoCaps items (None = all)
MAX_ITEMS     = None
# ─────────────────────────────────────────────────────────────────────────────


def backup(path: Path) -> Path:
    """Copy file to <name>.bak, return backup path."""
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"  Backed up → {bak.name}")
    return bak


def load_existing(index_file: Path, meta_file: Path):
    """Load existing FAISS index and metadata."""
    print(f"\n[1/5] Loading existing index...")
    print(f"  index : {index_file}")
    print(f"  meta  : {meta_file}")

    if not index_file.exists():
        raise FileNotFoundError(
            f"image.index not found at {index_file}\n"
            f"Check INDICES_DIR in the config block."
        )

    index = faiss.read_index(str(index_file))
    print(f"  Vectors in index : {index.ntotal}")
    print(f"  Dimension        : {index.d}")
    print(f"  Metric           : {'IP (cosine)' if index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'}")

    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"  Metadata entries : {len(metadata)}")
    else:
        print(f"  WARNING: {meta_file} not found — starting with empty metadata")
        metadata = []

    return index, metadata


def _load_nocaps_split(split: str):
    """
    Load a NoCaps split, automatically handling datasets 3.x vs 4.x.

    datasets >= 4.0 dropped support for custom loading scripts (NoCaps.py).
    We try the standard path first; if that fails we fall back to loading
    the auto-converted parquet files directly from HuggingFace, which works
    on ALL datasets versions and requires no downgrade.

    Parquet URL pattern (HF auto-converts every dataset):
        hf://datasets/HuggingFaceM4/NoCaps@refs/convert/parquet/
            default/<split>-*.parquet
    """
    from datasets import load_dataset
    import datasets as ds_lib

    ds_version = tuple(int(x) for x in ds_lib.__version__.split(".")[:2])

    if ds_version < (4, 0):
        # datasets 3.x — standard path works fine
        print(f"    (datasets {ds_lib.__version__} — using standard loader)")
        return load_dataset(
            "HuggingFaceM4/NoCaps",
            split=split,
            streaming=True,
        )

    # datasets 4.x — bypass the broken NoCaps.py script via parquet
    print(f"    (datasets {ds_lib.__version__} detected — using parquet fallback)")

    # HuggingFace auto-converts every dataset to parquet at this ref.
    # NoCaps split names in the parquet repo: "validation", "test"
    parquet_url = (
        f"hf://datasets/HuggingFaceM4/NoCaps"
        f"@refs/convert/parquet/default/{split}-*.parquet"
    )

    try:
        dataset = load_dataset(
            "parquet",
            data_files={split: parquet_url},
            split=split,
            streaming=True,
        )
        # Verify it has at least one expected column
        first = next(iter(dataset))
        if "annotations_captions" not in first and "image" not in first:
            raise ValueError("Parquet schema mismatch — unexpected columns")
        # Re-create the iterator (next() consumed the first row)
        dataset = load_dataset(
            "parquet",
            data_files={split: parquet_url},
            split=split,
            streaming=True,
        )
        return dataset

    except Exception as e:
        raise RuntimeError(
            f"Both standard and parquet loaders failed for NoCaps split='{split}'.\n"
            f"Parquet error: {e}\n\n"
            f"Fix options:\n"
            f"  1) pip install 'datasets==3.6.0'   (quickest)\n"
            f"  2) Set HF_TOKEN env var if the parquet ref requires auth\n"
            f"     $env:HF_TOKEN='hf_...'   (PowerShell)\n"
            f"     export HF_TOKEN='hf_...' (bash)"
        ) from e


def stream_and_embed(
    splits: list[str],
    clip_embedder,
    dim: int,
    existing_ids: set[str],
    max_items: int | None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Stream NoCaps from HuggingFace, embed with CLIP, return (vectors, meta).
    Handles datasets 3.x and 4.x automatically.
    """

    all_vectors = []
    all_meta    = []
    emitted     = 0

    for split in splits:
        split_tag = "val" if split == "validation" else "test"
        print(f"\n  Streaming split='{split}'...")

        dataset = _load_nocaps_split(split)

        batch_imgs  = []
        batch_meta  = []

        def flush_batch():
            nonlocal all_vectors, all_meta
            if not batch_imgs:
                return
            # embed_images accepts list of PIL images
            vecs = clip_embedder.embed_images(batch_imgs, batch_size=BATCH_SIZE)
            # Sanity-check dimension
            if vecs.shape[1] != dim:
                raise RuntimeError(
                    f"Embedding dim mismatch: got {vecs.shape[1]}, expected {dim}. "
                    f"Make sure CLIPEmbedder uses the same model as your existing index."
                )
            all_vectors.append(vecs)
            all_meta.extend(batch_meta)
            batch_imgs.clear()
            batch_meta.clear()

        # Tell PIL to load truncated/corrupted images instead of crashing.
        # Truncated images just have a corrupted bottom strip — safe to embed.
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        skipped = 0
        pbar = tqdm(dataset, desc=f"  [{split}]", unit="img")
        for idx, row in enumerate(pbar):
            if max_items and emitted >= max_items:
                break

            try:
                pil_img = row.get("image")
                if pil_img is None:
                    skipped += 1
                    continue

                # Force full pixel decode NOW so any truncation error
                # surfaces here (skippable) rather than inside the embedder.
                pil_img.load()

                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

            except (OSError, Exception):
                skipped += 1
                pbar.set_postfix(skipped=skipped)
                continue  # skip corrupted image, keep going

            captions  = row.get("annotations_captions") or []
            image_id  = str(row.get("image_id", idx))
            item_id   = f"nocaps_{split_tag}_{idx:06d}"

            # Skip if already in index (safe re-run)
            if item_id in existing_ids:
                continue

            batch_imgs.append(pil_img)

            # ── Metadata schema matching UnifiedSemanticIndex expectations ──
            # extract_semantic_content() checks: captions → caption → content → path
            batch_meta.append({
                "id"             : item_id,
                "captions"       : captions,          # list[str] — all 10
                "caption"        : captions[0] if captions else "",
                "content"        : captions[0] if captions else "",
                "path"           : "",                # no disk copy
                "source"         : "nocaps",
                "split"          : split,
                "nocaps_image_id": image_id,
                "url"            : row.get("image_coco_url", ""),
                "height"         : row.get("image_height", 0),
                "width"          : row.get("image_width", 0),
            })

            if len(batch_imgs) >= BATCH_SIZE:
                flush_batch()

            emitted += 1

        flush_batch()  # final partial batch
        if skipped:
            print(f"  Skipped {skipped} corrupted/missing images in [{split}]")
        split_count = len(all_meta) - sum(
            1 for m in all_meta if m.get("split") != split
        )
        print(f"  Embedded {emitted} NoCaps images so far  "
              f"(this split: {len([m for m in all_meta if m.get('split')==split])})")

    if not all_vectors:
        return np.empty((0, dim), dtype="float32"), []

    return np.vstack(all_vectors).astype("float32"), all_meta


def main():
    print("=" * 60)
    print("  DCASS — Add NoCaps to Image Index")
    print("=" * 60)

    # ── 1. Load existing index + metadata ────────────────────────────────────
    index, metadata = load_existing(INDEX_FILE, META_FILE)
    dim = index.d

    existing_ids = {m.get("id", "") for m in metadata}
    print(f"  Existing unique IDs: {len(existing_ids)}")

    # ── 2. Load CLIPEmbedder (same one your project uses) ────────────────────
    print(f"\n[2/5] Loading CLIPEmbedder...")
    from src.corpus.embedders.clip_embedder import CLIPEmbedder
    embedder = CLIPEmbedder()
    embedder._ensure_loaded()
    print(f"  Ready on device={embedder.device}")

    # ── 3. Stream + embed NoCaps ─────────────────────────────────────────────
    print(f"\n[3/5] Streaming & embedding NoCaps {NOCAPS_SPLITS}...")
    t0 = time.time()
    new_vectors, new_meta = stream_and_embed(
        NOCAPS_SPLITS, embedder, dim, existing_ids, MAX_ITEMS
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(new_meta)} new vectors")

    if len(new_meta) == 0:
        print("  Nothing new to add (all NoCaps items already in index). Exiting.")
        return

    # ── 4. Merge into existing index ─────────────────────────────────────────
    print(f"\n[4/5] Merging into index...")
    print(f"  Before : {index.ntotal:,} vectors")
    index.add(new_vectors)
    metadata.extend(new_meta)
    print(f"  After  : {index.ntotal:,} vectors  (+{len(new_meta):,})")
    print(f"  Metadata entries: {len(metadata):,}")

    # Sanity check
    assert index.ntotal == len(metadata), (
        f"MISMATCH: index has {index.ntotal} vectors but metadata has {len(metadata)} entries. "
        f"Aborting before save."
    )

    # ── 5. Backup originals + save ────────────────────────────────────────────
    print(f"\n[5/5] Saving...")
    if INDEX_FILE.exists():
        backup(INDEX_FILE)
    if META_FILE.exists():
        backup(META_FILE)

    faiss.write_index(index, str(INDEX_FILE))
    print(f"  Saved index    → {INDEX_FILE}  ({INDEX_FILE.stat().st_size / 1e6:.1f} MB)")

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"  Saved metadata → {META_FILE}  ({META_FILE.stat().st_size / 1e6:.1f} MB)")

    # ── Summary ───────────────────────────────────────────────────────────────
    nocaps_count = sum(1 for m in metadata if m.get("source") == "nocaps")
    flickr_count = len(metadata) - nocaps_count

    print(f"""
{'='*60}
  Merge complete!

  Flickr (original) : {flickr_count:,} images
  NoCaps (added)    : {nocaps_count:,} images
  Total in index    : {index.ntotal:,} vectors
  Dimension         : {dim}

  Files updated:
    {INDEX_FILE}
    {META_FILE}

  Originals backed up as .bak files.
{'='*60}
""")

    # ── Quick search sanity check ─────────────────────────────────────────────
    print("Quick sanity check — searching 'a dog in the park'...")
    query_vec = embedder.embed_text("a dog in the park").reshape(1, -1)
    D, I      = index.search(query_vec, k=3)
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        m = metadata[idx]
        src = m.get("source", "?")
        cap = m.get("caption", m.get("content", ""))[:70]
        print(f"  [{rank+1}] score={dist:.4f}  src={src:7s}  caption='{cap}'")


if __name__ == "__main__":
    main()