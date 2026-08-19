"""
src/stealth/stego_dispatch_demo.py

End-to-end demo: secret message → encode → GAN timing → dispatch log.

Run:
    cd <project_root>
    python -m src.stealth.stego_dispatch_demo
    python -m src.stealth.stego_dispatch_demo --secret "Three dogs on the beach"
    python -m src.stealth.stego_dispatch_demo --secret "..." --live    # real delays
    python -m src.stealth.stego_dispatch_demo --preview-only          # just show timing patterns
"""

from __future__ import annotations

import argparse
import json
import sys
import statistics
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))


def print_delay_analysis(sessions: list[list[float]], secret: str):
    """
    Print multi-session delay analysis to prove randomisation.

    This is the anti-analysis evidence: shows that no two sessions
    share the same pattern, making statistical fingerprinting impossible.
    """
    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  ANTI-ANALYSIS: DELAY PATTERN COMPARISON")
    print(f"  Secret: \"{secret}\"")
    print(SEP)
    print(f"  Each row = one independent session (same secret, different z~N(0,1))")
    print(f"  If patterns were identical → trivially detectable.")
    print(f"  High CV (>1.0) = irregular = hard to fingerprint.\n")

    n_carriers = len(sessions[0]) if sessions else 0
    header = f"  {'Session':<10}" + "".join(f"  d[{i+1}]" for i in range(n_carriers))
    print(header)
    print(f"  {'─'*10}" + "  ─────" * n_carriers)

    for s_idx, delays in enumerate(sessions):
        row = f"  Session {s_idx+1:<3}"
        for d in delays:
            row += f"  {d:>5.2f}"
        cv = statistics.stdev(delays) / statistics.mean(delays) if len(delays) > 1 and statistics.mean(delays) > 0 else 0
        row += f"   CV={cv:.2f}"
        print(row)

    print()
    all_delays = [d for s in sessions for d in s]
    print(f"  Overall stats across all sessions:")
    print(f"    mean  = {statistics.mean(all_delays):.3f}s")
    print(f"    stdev = {statistics.stdev(all_delays):.3f}s")
    overall_cv = statistics.stdev(all_delays) / statistics.mean(all_delays)
    print(f"    CV    = {overall_cv:.3f}  {'✓ high variance (stealthy)' if overall_cv > 0.5 else '⚠ low variance (needs training)'}")
    print(SEP)


def print_channel_analysis(result):
    """Show channel spreading — each carrier goes to a different channel."""
    print("\n  CHANNEL SPREAD (anti-burst-detection):")
    print(f"  {'Carrier':<8} {'Chunk':<25} {'Channel':<12} Delay")
    print(f"  {'─'*8} {'─'*25} {'─'*12} {'─'*8}")
    for e in result.events:
        chunk = e.chunk[:23] + "…" if len(e.chunk) > 23 else e.chunk
        print(f"  #{e.sequence_pos+1:<6} {chunk:<25} {e.channel_name:<12} {e.planned_delay:.3f}s")
    print()
    print(f"  Channel distribution: {result.channel_distribution}")


def run_dispatch_demo(
    secret      : str,
    dry_run     : bool = True,
    preview_only: bool = False,
    n_sessions  : int  = 5,
    generator_path: Path = None,
):
    from src.engine.encoder import SemanticEncoder
    from src.stealth.stealth_dispatcher import StealthDispatcher

    SEP = "=" * 65

    print(f"\n{SEP}")
    print("  DCASS — Stealth Dispatch with GAN Timing")
    print(SEP)
    print(f"  Secret      : \"{secret}\"")
    print(f"  Mode        : {'DRY RUN (fast)' if dry_run else 'LIVE (real delays)'}")
    print(SEP)

    # ── Init dispatcher ───────────────────────────────────────────────────────
    dispatcher = StealthDispatcher(
        generator_path = generator_path,
        device         = "cpu",
        scale_delays   = 1.0,
        dry_run_scale  = 0.001,   # 1000x faster in dry run
    )

    # ── Preview: show delay randomisation across sessions ────────────────────
    print(f"\n[1/{3 if not preview_only else 2}] Previewing delay patterns ({n_sessions} sessions)...")

    # Rough chunk count without loading full encoder
    import re
    delimiters = re.compile(
        r',\s*|\s+and\s+|\s+but\s+|\s+or\s+|\s+in\s+the\s+|\s+on\s+the\s+'
        r'|\s+at\s+the\s+|\s+with\s+the\s+|\s+of\s+the\s+|\s+near\s+the\s+',
        re.IGNORECASE
    )
    approx_chunks = max(1, len([p for p in delimiters.split(secret) if len(p.strip()) >= 3]))
    n_carriers    = max(2, approx_chunks)   # at least 2 for meaningful timing

    sessions = dispatcher.preview_delays(
        n_carriers  = n_carriers,
        n_sessions  = n_sessions,
        time_of_day = datetime.now().hour,
    )
    print_delay_analysis(sessions, secret)

    if preview_only:
        print("\n[2/2] Preview complete. Use --no-preview to run full encode+dispatch.")
        return

    # ── Encode the secret ─────────────────────────────────────────────────────
    print(f"\n[2/3] Encoding secret with SemanticEncoder...")
    encoder = SemanticEncoder(default_modalities=["image"])
    status  = encoder.load(modalities=["image"])

    if not any(status.values()):
        print("  ERROR: No index loaded. Run add_nocaps_to_index.py first.")
        return

    encoding = encoder.encode(
        secret,
        modalities       = ["image"],
        avoid_duplicates = True,
    )
    print(f"  Chunks     : {len(encoding.chunks)}")
    print(f"  Carriers   : {len(encoding.encoded)}")
    for i, enc in enumerate(encoding.encoded):
        print(f"    [{i+1}] \"{enc.chunk.original}\" → {enc.media.id} ({enc.media.normalized_score:.3f})")

    # ── Dispatch with GAN timing ──────────────────────────────────────────────
    print(f"\n[3/3] Dispatching with GAN-generated timing (dry_run={dry_run})...")
    print(f"  {'─'*63}")

    result = dispatcher.dispatch(
        encoding    = encoding,
        time_of_day = datetime.now().hour,
        dry_run     = dry_run,
        on_transmit = None,   # plug in your actual channel send here
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(result.summary())
    print_channel_analysis(result)

    # ── Save JSON log ─────────────────────────────────────────────────────────
    log = {
        "secret"          : result.secret,
        "session_start"   : result.session_start.isoformat(),
        "session_end"     : result.session_end.isoformat(),
        "latent_seed"     : result.latent_seed,
        "time_of_day_hour": result.time_of_day_hour,
        "anti_analysis"   : result.anti_analysis_score,
        "delay_stats"     : result.delay_stats,
        "channel_dist"    : result.channel_distribution,
        "events"          : [
            {
                "pos"          : e.sequence_pos,
                "carrier_id"   : e.carrier_id,
                "chunk"        : e.chunk,
                "caption"      : e.caption,
                "channel"      : e.channel_name,
                "planned_delay": round(e.planned_delay, 4),
                "actual_delay" : round(e.actual_delay, 4),
                "confidence"   : round(e.confidence, 4),
                "source"       : e.source,
            }
            for e in result.events
        ],
        "delay_sequences" : sessions,   # all preview sessions for anti-analysis proof
    }

    out_path = _ROOT / "stego_dispatch_log.json"
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Dispatch log saved → {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="DCASS Stealth Dispatch Demo")
    parser.add_argument("--secret",       type=str, default="Three dogs on the beach")
    parser.add_argument("--live",         action="store_true", help="Use real delays (no dry_run_scale)")
    parser.add_argument("--preview-only", action="store_true", help="Only show timing patterns, skip encode")
    parser.add_argument("--sessions",     type=int, default=5, help="Preview sessions to show (default 5)")
    parser.add_argument("--gen-path",     type=str, default=None, help="Path to trained generator checkpoint")
    args = parser.parse_args()

    run_dispatch_demo(
        secret         = args.secret,
        dry_run        = not args.live,
        preview_only   = args.preview_only,
        n_sessions     = args.sessions,
        generator_path = Path(args.gen_path) if args.gen_path else None,
    )


if __name__ == "__main__":
    main()