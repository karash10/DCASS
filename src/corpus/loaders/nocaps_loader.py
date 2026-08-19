"""
corpus/loaders/nocaps_loader.py

NoCaps Dataset Loader — drop-in replacement for FlickrLoader.

Streams images + captions from HuggingFace (HuggingFaceM4/NoCaps)
with ZERO images saved to disk. Implements the exact same interface
as FlickrLoader so UnifiedSemanticIndex / encoder / decoder see no
difference between Flickr and NoCaps items.

Schema produced per item (identical to FlickrLoader output):
    {
        "id"         : "nocaps_val_000042",
        "image_path" : "",                     # empty — no disk copy
        "content"    : "",                     # empty — streaming only
        "caption"    : "a dog on a bench",     # first of 10 captions
        "captions"   : ["...", "...", ...],    # all 10 captions
        "pil_image"  : <PIL.Image>,            # in-memory only
        "source"     : "nocaps",
        "split"      : "validation",
        "url"        : "https://s3.../x.jpg",
        "height"     : 480,
        "width"      : 640,
    }

Usage:
    loader = NoCapsLoader(splits=["validation"])
    for item in loader.load():
        emb = clip_embedder.embed_image(item["pil_image"])
        ...

NOTE: pil_image is a PIL.Image.RGB held in RAM only during iteration.
      It is NOT persisted — garbage collected after each batch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

from .base_loader import BaseLoader


class NoCapsLoader(BaseLoader):
    """
    Loader for the NoCaps dataset (HuggingFaceM4/NoCaps).

    Streams images + captions directly from HuggingFace — no files
    written to disk. Fully compatible with FlickrLoader's output schema
    so the rest of the DCASS pipeline (embedders, indexer, encoder,
    decoder) works without any modification.

    Args:
        source_path : Dummy path (BaseLoader requires one). Pass any Path.
        splits      : NoCaps splits to load. NoCaps has no train split.
                      Options: ["validation"], ["test"], or both.
        max_items   : Optional cap on total items (useful for testing).

    Example:
        loader = NoCapsLoader(Path("."), splits=["validation"])
        for item in loader.load():
            pil_img  = item["pil_image"]   # PIL.Image.RGB
            captions = item["captions"]    # list[str], up to 10
            item_id  = item["id"]          # "nocaps_val_000042"
    """

    VALID_SPLITS = ("validation", "test")

    def __init__(
        self,
        source_path: Path = Path("."),
        splits: List[str] = None,
        max_items: Optional[int] = None,
    ):
        super().__init__(Path(source_path))
        self.splits    = splits or ["validation"]
        self.max_items = max_items

        # Validate
        for s in self.splits:
            if s not in self.VALID_SPLITS:
                raise ValueError(
                    f"Invalid split '{s}'. NoCaps only has: {self.VALID_SPLITS}"
                )

        self._total_loaded = 0

    # ── BaseLoader interface ──────────────────────────────────────────────────

    @property
    def modality(self) -> str:
        return "image"

    def __len__(self) -> int:
        """
        Approximate count without streaming.
        validation=4500, test=10600 (official NoCaps numbers).
        """
        counts = {"validation": 4500, "test": 10600}
        total  = sum(counts.get(s, 0) for s in self.splits)
        if self.max_items:
            total = min(total, self.max_items)
        return total

    def load(self) -> Iterator[Dict[str, Any]]:
        """
        Stream NoCaps items one by one.

        Yields dicts that are schema-compatible with FlickrLoader.load().
        The extra keys (pil_image, source, split, url, height, width)
        are used by add_nocaps_to_index.py but ignored by everything
        else in the pipeline.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace `datasets` library is required for NoCapsLoader.\n"
                "Install with:  pip install datasets"
            )

        emitted = 0

        for split in self.splits:
            split_tag = "val" if split == "validation" else "test"

            print(f"[NoCapsLoader] Streaming split='{split}' from HuggingFace...")
            dataset = load_dataset(
                "HuggingFaceM4/NoCaps",
                split=split,
                streaming=True,     # ← key: no files written to disk
            )

            for idx, row in enumerate(dataset):
                if self.max_items and emitted >= self.max_items:
                    return

                pil_image = row.get("image")
                if pil_image is None:
                    continue

                # Normalise to RGB (some Open Images are grayscale / RGBA)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                captions  = row.get("annotations_captions") or []
                image_id  = str(row.get("image_id", idx))
                item_id   = f"nocaps_{split_tag}_{idx:06d}"

                yield {
                    # ── FlickrLoader-compatible keys ──────────────────────
                    "id"         : item_id,
                    "image_path" : "",           # no disk path
                    "content"    : "",           # no disk path
                    "caption"    : captions[0] if captions else "",
                    "captions"   : captions,     # all 10 human captions
                    # ── NoCaps-specific extras ────────────────────────────
                    "pil_image"  : pil_image,    # PIL.Image in RAM only
                    "source"     : "nocaps",
                    "split"      : split,
                    "nocaps_image_id": image_id,
                    "url"        : row.get("image_coco_url", ""),
                    "height"     : row.get("image_height", 0),
                    "width"      : row.get("image_width", 0),
                }

                emitted += 1
                self._total_loaded = emitted

        print(f"[NoCapsLoader] Done — yielded {emitted} items.")
