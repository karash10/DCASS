"""
src/stealth/stealth_dispatcher.py

StealthDispatcher — integrates the SemanticEncoder with the GAN
to schedule carrier transmissions using dynamically generated,
human-like timing patterns.

Flow:
    secret message
        → SemanticEncoder   (message → carrier media IDs)
        → TemporalPatternGenerator  (carrier count → delay schedule)
        → StealthDispatcher (wait delay[i] → transmit carrier[i])
        → TransmissionLog   (audit trail with real timestamps)

Why this defeats advanced stego analysis:
    1. No fixed interval — every message gets a fresh z ~ N(0,1),
       so the delay pattern is unique per transmission session.
    2. GRU autocorrelation — delays are NOT i.i.d. They have burst
       patterns (short delays) followed by pauses, mimicking how
       humans actually post on social media.
    3. Time-of-day conditioning — a message sent at 2am gets slower,
       sparser delays than one sent at noon. This matches real human
       activity rhythms that DPI systems learn.
    4. Attention coherence — long sequences have global temporal
       structure, not just local. A burst at position 3 affects
       what happens at position 10.
    5. Channel diversity — carriers are spread across channels
       (e.g. different social platforms) so no single channel
       shows a suspicious burst pattern.
"""

from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import torch

from src.stealth.gan.generator import TemporalPatternGenerator, sample_latent, TimingSchedule
from src.engine.encoder import EncodingResult, EncodedChunk
from src.corpus.index.unified_index import MediaItem


# ── Channel names (maps channel index → human-readable label) ────────────────
CHANNEL_NAMES = {
    0: "primary",    # e.g. main social feed
    1: "secondary",  # e.g. stories / reels
    2: "tertiary",   # e.g. direct messages
}


@dataclass
class DispatchEvent:
    """
    A single carrier transmission event.

    Records everything about one carrier being sent:
    what was sent, when, on which channel, with what delay,
    and what the GAN's confidence was at that point.
    """
    sequence_pos  : int             # position in carrier sequence (0-indexed)
    carrier_id    : str             # media item ID (e.g. 'nocaps_val_000042')
    caption       : str             # semantic content of the carrier
    chunk         : str             # the secret chunk this carrier encodes
    channel       : int             # channel index (0,1,2)
    channel_name  : str             # human-readable channel label
    planned_delay : float           # delay in seconds (from GAN)
    actual_delay  : float           # real elapsed time (may differ slightly)
    scheduled_at  : datetime        # wall-clock time when dispatch was triggered
    confidence    : float           # GAN generator confidence at this position
    source        : str             # 'flickr' or 'nocaps'


@dataclass
class DispatchResult:
    """
    Complete result of a stealthy transmission session.

    Contains the full transmission log, timing statistics,
    and anti-analysis evidence.
    """
    secret          : str
    events          : list[DispatchEvent]
    session_start   : datetime
    session_end     : datetime
    latent_seed     : int           # random seed used — reproducible if needed
    time_of_day_hour: int           # hour used for GAN conditioning

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def total_duration_s(self) -> float:
        """Total wall time for the full transmission."""
        delta = self.session_end - self.session_start
        return delta.total_seconds()

    @property
    def planned_delays(self) -> list[float]:
        return [e.planned_delay for e in self.events]

    @property
    def actual_delays(self) -> list[float]:
        return [e.actual_delay for e in self.events]

    @property
    def channel_distribution(self) -> dict[str, int]:
        """How many carriers were sent on each channel."""
        dist: dict[str, int] = {}
        for e in self.events:
            dist[e.channel_name] = dist.get(e.channel_name, 0) + 1
        return dist

    @property
    def delay_stats(self) -> dict[str, float]:
        """Descriptive stats of the delay sequence."""
        import statistics
        d = self.planned_delays
        if not d:
            return {}
        return {
            "min_s"   : round(min(d), 3),
            "max_s"   : round(max(d), 3),
            "mean_s"  : round(statistics.mean(d), 3),
            "stdev_s" : round(statistics.stdev(d), 3) if len(d) > 1 else 0.0,
            "cv"      : round(                             # coefficient of variation
                statistics.stdev(d) / statistics.mean(d), 3
            ) if len(d) > 1 and statistics.mean(d) > 0 else 0.0,
        }

    @property
    def anti_analysis_score(self) -> float:
        """
        Heuristic score [0,1] estimating how hard this pattern is to detect.

        Higher is stealthier. Based on:
          - High CV (coefficient of variation) of delays → irregular = good
          - Channel diversity → spread = good
          - Avoidance of too-short delays (< 1s looks robotic)
        """
        stats = self.delay_stats
        if not stats:
            return 0.0

        cv_score       = min(1.0, stats["cv"] / 1.5)           # penalise low CV
        diversity      = len(self.channel_distribution) / 3.0  # 3 channels max
        min_delay_ok   = 1.0 if stats["min_s"] >= 1.0 else 0.5
        return round((cv_score * 0.5 + diversity * 0.3 + min_delay_ok * 0.2), 3)

    def summary(self) -> str:
        stats = self.delay_stats
        lines = [
            "=" * 60,
            "  STEALTH DISPATCH RESULT",
            "=" * 60,
            f"  Secret         : \"{self.secret}\"",
            f"  Carriers sent  : {len(self.events)}",
            f"  Total time     : {self.total_duration_s:.2f}s",
            f"  Session start  : {self.session_start.strftime('%H:%M:%S')}",
            f"  Time-of-day    : {self.time_of_day_hour:02d}:00",
            f"  Latent seed    : {self.latent_seed}",
            "",
            "  Delay statistics (GAN-generated):",
            f"    min  = {stats.get('min_s',0):.3f}s",
            f"    max  = {stats.get('max_s',0):.3f}s",
            f"    mean = {stats.get('mean_s',0):.3f}s",
            f"    σ    = {stats.get('stdev_s',0):.3f}s",
            f"    CV   = {stats.get('cv',0):.3f}  (>1 = highly irregular)",
            "",
            "  Channel distribution:",
        ]
        for ch, count in self.channel_distribution.items():
            lines.append(f"    {ch:12s}: {count} carrier(s)")
        lines += [
            "",
            f"  Anti-analysis score: {self.anti_analysis_score:.3f} / 1.000",
            "",
            "  Transmission sequence:",
            f"  {'#':<4} {'Delay':>7}  {'Ch':>10}  {'Carrier ID':>22}  Caption",
            f"  {'─'*4} {'─'*7}  {'─'*10}  {'─'*22}  {'─'*35}",
        ]
        for e in self.events:
            cap = e.caption[:35] + "…" if len(e.caption) > 35 else e.caption
            lines.append(
                f"  {e.sequence_pos+1:<4} {e.planned_delay:>7.3f}s"
                f"  {e.channel_name:>10}  {e.carrier_id:>22}  {cap}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Callback type ─────────────────────────────────────────────────────────────
OnTransmitCallback = Callable[[DispatchEvent], None]


class StealthDispatcher:
    """
    Connects the SemanticEncoder to the GAN timing scheduler.

    Given an EncodingResult (from SemanticEncoder.encode()), generates
    a human-like transmission schedule using TemporalPatternGenerator
    and dispatches each carrier with the appropriate delay.

    Usage:
        dispatcher = StealthDispatcher()

        # Encode a secret
        result = encoder.encode("Three dogs on the beach")

        # Dispatch with GAN-generated timing
        dispatch = dispatcher.dispatch(result, dry_run=True)
        print(dispatch.summary())

    Args:
        generator       : Pre-loaded TemporalPatternGenerator (loaded if None)
        generator_path  : Path to saved generator .pt checkpoint
        device          : torch device
        scale_delays    : Multiply GAN delays by this factor (default 1.0)
                          Use 0.01 for testing to avoid real waits
        dry_run_scale   : Scale factor applied when dry_run=True (default 0.001)
    """

    def __init__(
        self,
        generator         : Optional[TemporalPatternGenerator] = None,
        generator_path    : Optional[Path] = None,
        device            : str = "cpu",
        scale_delays      : float = 1.0,
        dry_run_scale     : float = 0.001,
    ):
        self.device        = device
        self.scale_delays  = scale_delays
        self.dry_run_scale = dry_run_scale
        self._generator    = generator
        self._gen_path     = generator_path

    # ── Lazy load ─────────────────────────────────────────────────────────────

    @property
    def generator(self) -> TemporalPatternGenerator:
        if self._generator is None:
            self._generator = TemporalPatternGenerator(
                latent_dim          = 128,
                hidden_dim          = 256,
                num_channels        = 3,
                max_sequence_length = 100,
            ).to(self.device)
            self._generator.eval()

            if self._gen_path and Path(self._gen_path).exists():
                ckpt = torch.load(self._gen_path, map_location=self.device)
                # Support both raw state_dict and trainer checkpoint
                state = ckpt.get("generator_state", ckpt)
                self._generator.load_state_dict(state)
                print(f"[Dispatcher] Loaded generator from {self._gen_path}")
            else:
                print("[Dispatcher] Using untrained generator "
                      "(run GANTrainer first for production use)")
        return self._generator

    # ── Core dispatch ─────────────────────────────────────────────────────────

    def _generate_schedule(
        self,
        n_carriers   : int,
        time_of_day  : Optional[int] = None,
        seed         : Optional[int] = None,
    ) -> tuple[TimingSchedule, int, int]:
        """
        Call the GAN to produce a TimingSchedule for n_carriers.

        Returns (schedule, seed_used, hour_used).
        """
        if seed is not None:
            torch.manual_seed(seed)
        actual_seed = torch.initial_seed() % (2**31)

        hour = time_of_day if time_of_day is not None else datetime.now().hour

        z            = sample_latent(1, latent_dim=128, device=self.device)
        tod_tensor   = torch.tensor([float(hour)], device=self.device)

        with torch.no_grad():
            schedule = self.generator(z, sequence_length=n_carriers, time_of_day=tod_tensor)

        return schedule, actual_seed, hour

    def dispatch(
        self,
        encoding        : EncodingResult,
        time_of_day     : Optional[int]             = None,
        seed            : Optional[int]             = None,
        dry_run         : bool                      = True,
        on_transmit     : Optional[OnTransmitCallback] = None,
    ) -> DispatchResult:
        """
        Dispatch carriers from an EncodingResult with GAN-generated timing.

        Args:
            encoding    : Output of SemanticEncoder.encode()
            time_of_day : Hour 0–23 for GAN conditioning (uses current hour if None)
            seed        : Random seed for reproducibility (random if None)
            dry_run     : If True, apply dry_run_scale to delays (fast testing)
            on_transmit : Optional callback fired after each carrier is sent

        Returns:
            DispatchResult with full transmission log
        """
        carriers     = encoding.encoded
        n            = len(carriers)

        if n == 0:
            raise ValueError("EncodingResult has no encoded carriers.")
        if n > self.generator.max_sequence_length:
            raise ValueError(
                f"Too many carriers ({n}) for generator "
                f"(max {self.generator.max_sequence_length})."
            )

        # ── Generate timing schedule from GAN ─────────────────────────────────
        schedule, actual_seed, hour = self._generate_schedule(n, time_of_day, seed)

        delays_raw  = schedule.delays[0].cpu().tolist()           # (n,)  float seconds
        channels    = schedule.sample_channels()[0].cpu().tolist() # (n,)  int 0,1,2
        confidence  = float(schedule.confidence[0].cpu())

        # Apply scale
        scale       = self.scale_delays * (self.dry_run_scale if dry_run else 1.0)
        delays      = [max(0.001, d * scale) for d in delays_raw]

        # ── Transmit sequence ─────────────────────────────────────────────────
        events       : list[DispatchEvent] = []
        session_start = datetime.now(timezone.utc)

        for i, enc_chunk in enumerate(carriers):
            t_before = time.perf_counter()

            # Wait for the GAN-scheduled delay
            time.sleep(delays[i])

            t_after     = time.perf_counter()
            actual_wait = t_after - t_before

            media   = enc_chunk.media
            ch_idx  = int(channels[i]) % 3
            caption = (
                media.metadata.get("captions", [None])[0]
                or media.metadata.get("caption", "")
                or media.content
                or ""
            )

            event = DispatchEvent(
                sequence_pos  = i,
                carrier_id    = media.id,
                caption       = caption,
                chunk         = enc_chunk.chunk.original,
                channel       = ch_idx,
                channel_name  = CHANNEL_NAMES[ch_idx],
                planned_delay = delays_raw[i],   # raw GAN output (unscaled) for analysis
                actual_delay  = actual_wait / scale if scale > 0 else actual_wait,
                scheduled_at  = datetime.now(timezone.utc),
                confidence    = confidence,
                source        = media.metadata.get("source", "flickr"),
            )
            events.append(event)

            # Fire callback (e.g. actually post to a channel)
            if on_transmit:
                on_transmit(event)

            self._print_event(event, dry_run)

        session_end = datetime.now(timezone.utc)

        return DispatchResult(
            secret           = encoding.original_message,
            events           = events,
            session_start    = session_start,
            session_end      = session_end,
            latent_seed      = actual_seed,
            time_of_day_hour = hour,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _print_event(self, e: DispatchEvent, dry_run: bool):
        tag  = "[DRY RUN] " if dry_run else "[LIVE]    "
        cap  = e.caption[:50] + "…" if len(e.caption) > 50 else e.caption
        print(
            f"  {tag}#{e.sequence_pos+1:02d}  "
            f"delay={e.planned_delay:8.3f}s  "
            f"ch={e.channel_name:10s}  "
            f"id={e.carrier_id:25s}  \"{cap}\""
        )

    def preview_delays(
        self,
        n_carriers   : int,
        n_sessions   : int = 5,
        time_of_day  : Optional[int] = None,
    ) -> list[list[float]]:
        """
        Generate multiple delay schedules WITHOUT dispatching.

        Useful for showing that each session gets a unique, irregular
        delay pattern — anti-analysis evidence.

        Args:
            n_carriers  : Number of carriers to schedule
            n_sessions  : How many independent sessions to preview
            time_of_day : Hour for conditioning (current hour if None)

        Returns:
            List of delay sequences (one per session)
        """
        sessions = []
        for _ in range(n_sessions):
            sched, _, _ = self._generate_schedule(n_carriers, time_of_day)
            delays = sched.delays[0].cpu().tolist()
            sessions.append([round(d, 3) for d in delays])
        return sessions