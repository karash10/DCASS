"""
src/stealth/stego_demo.py

Interactive semantic steganography demo for DCASS.

Run:
    cd <project_root>
    python -m src.stealth.stego_demo
    
Or with a specific secret:
    python -m src.stealth.stego_demo --secret "Three dogs on the beach"

What it shows:
  1. Input secret message
  2. Semantic chunks produced by SemanticChunker
  3. Top-K image candidates per chunk (FAISS search)
  4. Selected carrier images with similarity scores
  5. Decoded captions from each carrier
  6. Reconstructed meaning vs original
  7. Accuracy metrics (BLEU-1, semantic cosine)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))


def compute_bleu1(reference: str, hypothesis: str) -> float:
    """BLEU-1 without nltk dependency."""
    ref_tokens  = set(reference.lower().split())
    hyp_tokens  = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    matches = sum(1 for t in hyp_tokens if t in ref_tokens)
    return matches / len(hyp_tokens)


def compute_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_demo(secret: str, top_k: int = 3, image_only: bool = True):
    from src.corpus.index.unified_index import UnifiedSemanticIndex
    from src.engine.chunker import SemanticChunker
    import clip, torch

    SEP  = "=" * 60
    SEP2 = "-" * 60

    print(f"\n{SEP}")
    print("  DCASS — Dynamic Semantic Steganography Demo")
    print(SEP)
    print(f"  Secret : \"{secret}\"")
    print(f"  Mode   : images only  |  top-{top_k} candidates per chunk")
    print(SEP)

    # ── Load index ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading index...")
    index = UnifiedSemanticIndex()
    status = index.load(modalities=["image"])
    if not status.get("image"):
        print("  ERROR: image index not loaded. Run add_nocaps_to_index.py first.")
        return None

    n_flickr = sum(1 for m in index.metadata["image"] if m.get("source","flickr") != "nocaps")
    n_nocaps = sum(1 for m in index.metadata["image"] if m.get("source") == "nocaps")
    print(f"  Total vectors : {index.indices['image'].ntotal:,}")
    print(f"  Flickr        : {n_flickr:,}")
    print(f"  NoCaps        : {n_nocaps:,}")

    # ── Chunk the secret ─────────────────────────────────────────────────────
    print(f"\n[2/4] Chunking secret...")
    chunker = SemanticChunker(expand_synonyms=False)
    chunks  = chunker.chunk(secret)
    print(f"  Chunks ({len(chunks)}):")
    for c in chunks:
        print(f"    [{c.index}] \"{c.original}\"")

    # ── FAISS search per chunk ────────────────────────────────────────────────
    print(f"\n[3/4] Searching FAISS index...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    meta_list = index.metadata["image"]
    faiss_idx = index.indices["image"]

    results_per_chunk = []

    for chunk in chunks:
        with torch.no_grad():
            tokens = clip.tokenize([chunk.text], truncate=True).to(device)
            vec    = model.encode_text(tokens)
            vec    = vec / vec.norm(dim=-1, keepdim=True)
            q      = vec.cpu().float().numpy()

        scores, idxs = faiss_idx.search(q, k=top_k * 4)

        candidates = []
        seen_ids   = set()
        for score, i in zip(scores[0], idxs[0]):
            if i < 0 or i >= len(meta_list):
                continue
            m = meta_list[i]
            mid = m.get("id", str(i))
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            caption = ""
            if m.get("captions"):
                caption = m["captions"][0]
            elif m.get("caption"):
                caption = m["caption"]
            elif m.get("content"):
                caption = m["content"]

            candidates.append({
                "faiss_idx" : i,
                "id"        : mid,
                "score"     : float(score),
                "caption"   : caption,
                "source"    : m.get("source", "flickr"),
                "url"       : m.get("url", ""),
                "path"      : m.get("path", ""),
            })
            if len(candidates) >= top_k:
                break

        results_per_chunk.append({
            "chunk"      : chunk.original,
            "candidates" : candidates,
            "selected"   : candidates[0] if candidates else None,
        })

    # ── Print encode result ───────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  ENCODE RESULT")
    print(SEP2)
    carrier_captions = []

    for i, r in enumerate(results_per_chunk):
        print(f"\n  Chunk {i+1}: \"{r['chunk']}\"")
        print(f"  {'Rank':<5} {'Score':>7}  {'Source':>7}  Caption")
        print(f"  {'─'*5} {'─'*7}  {'─'*7}  {'─'*50}")

        for rank, c in enumerate(r["candidates"]):
            marker = "→" if rank == 0 else " "
            cap    = (c["caption"][:55] + "…") if len(c["caption"]) > 55 else c["caption"]
            src    = c["source"][:7]
            print(f"  {marker} #{rank+1:<3} {c['score']:>7.4f}  {src:>7}  {cap}")

        if r["selected"]:
            carrier_captions.append(r["selected"]["caption"])

    # ── Decode ────────────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  DECODE RESULT")
    print(SEP2)
    reconstructed = " | ".join(carrier_captions)
    print(f"\n  Original      : \"{secret}\"")
    print(f"  Reconstructed : \"{reconstructed}\"")

    # ── Metrics ───────────────────────────────────────────────────────────────
    bleu1 = compute_bleu1(secret, reconstructed)

    with torch.no_grad():
        orig_tok  = clip.tokenize([secret],        truncate=True).to(device)
        rec_tok   = clip.tokenize([reconstructed], truncate=True).to(device)
        orig_emb  = model.encode_text(orig_tok).cpu().float().numpy()[0]
        rec_emb   = model.encode_text(rec_tok).cpu().float().numpy()[0]
    cos_sim = compute_cosine(orig_emb, rec_emb)

    print(f"\n  BLEU-1        : {bleu1:.4f}")
    print(f"  Semantic sim  : {cos_sim:.4f}  (CLIP cosine, 1.0 = identical)")

    # ── Carrier image URLs for display ───────────────────────────────────────
    print(f"\n{SEP2}")
    print("  CARRIER IMAGES")
    print(SEP2)
    output = {
        "secret"        : secret,
        "chunks"        : [r["chunk"] for r in results_per_chunk],
        "carriers"      : [],
        "reconstructed" : reconstructed,
        "metrics"       : {"bleu1": round(bleu1, 4), "semantic_cosine": round(cos_sim, 4)},
    }
    for i, r in enumerate(results_per_chunk):
        s = r["selected"]
        if not s:
            continue
        url  = s.get("url") or ""
        path = s.get("path") or ""
        print(f"\n  Chunk {i+1}: \"{r['chunk']}\"")
        print(f"    Caption : {s['caption']}")
        print(f"    Score   : {s['score']:.4f}")
        print(f"    Source  : {s['source']}")
        if url:
            print(f"    URL     : {url}")
        if path and Path(path).exists():
            print(f"    Path    : {path}")
        output["carriers"].append({
            "chunk"  : r["chunk"],
            "id"     : s["id"],
            "caption": s["caption"],
            "score"  : round(s["score"], 4),
            "source" : s["source"],
            "url"    : url,
            "path"   : path,
        })

    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  Secret split into {len(chunks)} chunk(s)")
    print(f"  {len(output['carriers'])} carrier image(s) selected")
    print(f"  BLEU-1 = {bleu1:.4f}  |  Semantic similarity = {cos_sim:.4f}")

    out_file = _ROOT / "stego_output.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_file}")
    print(SEP)

    return output


def main():
    parser = argparse.ArgumentParser(description="DCASS Semantic Stego Demo")
    parser.add_argument("--secret", type=str,
                        default="Three dogs on the beach",
                        help="Secret message to encode")
    parser.add_argument("--topk", type=int, default=3,
                        help="Candidates to show per chunk (default: 3)")
    args = parser.parse_args()
    run_demo(args.secret, top_k=args.topk)


if __name__ == "__main__":
    main()
