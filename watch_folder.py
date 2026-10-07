"""
watch_folder.py - CryptoStego Unattended Automation
==================================================
Monitor a directory for incoming digital images and automatically
encrypt + embed them into stego WAV files.  Every job is recorded
in a JSON-Lines audit log — useful for compliance and traceability
in any organization handling sensitive images.

Usage (CLI flags):
    python watch_folder.py \
        --input-dir  /incoming/images \
        --output-dir /outgoing/stego \
        --audio      /assets/cover_speech.wav \
        --key        "organization-secret-key" \
        --log        cryptostego_audit.log

Usage (JSON config file):
    python watch_folder.py --config cryptostego_watch.json

Config file format (cryptostego_watch.json):
    {
        "input_dir":  "/incoming/images",
        "output_dir": "/outgoing/stego",
        "audio":      "/assets/cover_speech.wav",
        "passphrase": "organization-secret-key",
        "log":        "cryptostego_audit.log"
    }
    "audio" is optional; omit it to use synthetic speech (demo only).

Requirements:
    pip install watchdog
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.io import wavfile

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _WD_OK = True
except ImportError:
    _WD_OK = False

# ── Backend ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_stego
import metrics
import image_io as _image_io
from pipeline import sender as _sender, _synth_audio, _payload_size

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".dcm"}


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_logger(log_path: str) -> logging.Logger:
    """
    Two handlers:
      - Console: human-readable timestamped lines
      - File:    JSON-Lines (one JSON object per line) for audit tools
    """
    logger = logging.getLogger("cryptostego.watch")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))   # raw JSON lines
    logger.addHandler(fh)
    return logger


def _audit(logger: logging.Logger, record: dict):
    """Append a structured record to the audit log."""
    record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    logger.info(json.dumps(record, default=str))


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE / AUDIO LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_image(path: str) -> np.ndarray:
    """
    Load any supported image preserving original bit depth and channel count.
    Uses image_io.load() for consistent behaviour with the GUI pipeline.
    No metadata or PHI is retained in the returned array.
    """
    arr, _ = _image_io.load(path)
    return arr


def _wait_stable(path: str, interval: float = 0.4, retries: int = 8) -> bool:
    """Poll until the file size stabilises (i.e. the writer has finished)."""
    prev = -1
    for _ in range(retries):
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size == prev > 0:
            return True
        prev = size
        time.sleep(interval)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHDOG EVENT HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class _ImageHandler(FileSystemEventHandler):
    """
    Collects newly created image files and processes them in batches.
    A 1.5-second settling window groups files that arrive together
    (e.g. a patient folder being copied all at once).
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self._cfg    = config
        self._log    = logger
        self._queue  = []
        self._last_t = 0.0

    # ── watchdog callback ────────────────────────────────────────────────────

    def on_created(self, event):
        if event.is_directory:
            return
        if os.path.splitext(event.src_path)[1].lower() in SUPPORTED_EXTS:
            self._queue.append(event.src_path)
            self._last_t = time.time()

    # ── called by main loop ──────────────────────────────────────────────────

    def poll(self):
        """Flush the queue when at least 1.5 s have passed since the last arrival."""
        if not self._queue:
            return
        if time.time() - self._last_t < 1.5:
            return      # still accumulating
        batch = list(self._queue)
        self._queue.clear()
        self._process_batch(batch)

    # ── batch processing ─────────────────────────────────────────────────────

    def _process_batch(self, paths: list):
        cfg = self._cfg
        log = self._log
        job = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        log.info(f"[{job}] Processing batch of {len(paths)} file(s) ...")

        # 1. Load images
        images, loaded = [], []
        for p in paths:
            if not _wait_stable(p):
                _audit(log, {
                    "event": "SKIP_UNSTABLE", "job": job, "path": p,
                    "reason": "File size did not stabilise in time.",
                })
                log.warning(f"  Skipped (unstable): {p}")
                continue
            try:
                img = _load_image(p)
                images.append(img)
                loaded.append(p)
                log.debug(f"  Loaded: {p}  {img.shape}")
            except Exception as exc:
                _audit(log, {
                    "event": "LOAD_ERROR", "job": job,
                    "path": p, "error": str(exc),
                })
                log.error(f"  Load error {p}: {exc}")

        if not images:
            log.warning(f"[{job}] No usable images in batch — nothing to do.")
            return

        # 2. Load cover audio
        try:
            if cfg.get("audio"):
                sr, audio = wavfile.read(cfg["audio"])
                if audio.ndim > 1:
                    audio = audio[:, 0]
                audio = audio.astype(np.int16)
            else:
                sr    = 16000
                audio = _synth_audio(60.0, sr)
        except Exception as exc:
            _audit(log, {"event": "AUDIO_ERROR", "job": job, "error": str(exc)})
            log.error(f"[{job}] Audio load failed: {exc}")
            return

        # 3. Capacity pre-check
        cap    = audio_stego.capacity_bits(audio)
        needed = _payload_size(images) * 8

        if needed > cap:
            _audit(log, {
                "event":       "CAPACITY_EXCEEDED",
                "job":         job,
                "needed_bits": needed,
                "cap_bits":    cap,
                "n_images":    len(images),
                "suggestion":  "Use a longer cover audio file or split the batch.",
            })
            log.error(
                f"[{job}] Capacity exceeded: need {needed:,} bits, "
                f"have {cap:,} bits. Increase audio length."
            )
            return

        # 4. Encrypt + embed
        try:
            _, stego, _ = _sender(images, audio, cfg["passphrase"])
        except Exception as exc:
            _audit(log, {"event": "ENCRYPT_ERROR", "job": job, "error": str(exc)})
            log.error(f"[{job}] Encryption failed: {exc}")
            return

        # 5. Write stego WAV to output directory
        os.makedirs(cfg["output_dir"], exist_ok=True)
        out_path = os.path.join(cfg["output_dir"], f"stego_{job}.wav")
        wavfile.write(out_path, sr, stego)

        snr_db  = metrics.snr(audio, stego)
        cap_pct = needed / cap * 100

        _audit(log, {
            "event":        "JOB_COMPLETE",
            "job":          job,
            "n_images":     len(images),
            "input_files":  loaded,
            "output_file":  out_path,
            "cap_pct":      round(cap_pct, 2),
            "snr_db":       round(snr_db, 2),
        })
        log.info(
            f"[{job}] DONE  {len(images)} image(s) -> {out_path} "
            f"(cap {cap_pct:.1f}%, SNR {snr_db:.1f} dB)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not _WD_OK:
        print(
            "ERROR: watchdog library is not installed.\n"
            "       pip install watchdog"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="CryptoStego watch-folder — automatically encrypt images into stego WAVs"
    )
    parser.add_argument("--config",     default=None,
                        help="JSON config file path (see module docstring for format)")
    parser.add_argument("--input-dir",  default=None,
                        help="Directory to monitor for new images")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write stego WAV files into")
    parser.add_argument("--audio",      default=None,
                        help="Cover speech WAV path (omit for synthetic demo audio)")
    parser.add_argument("--key",        default=None,
                        help="Encryption passphrase")
    parser.add_argument("--log",        default="cryptostego_audit.log",
                        help="Audit log file path (default: cryptostego_audit.log)")
    args = parser.parse_args()

    # Start from config file, then layer CLI overrides
    cfg: dict = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)

    if args.input_dir:  cfg["input_dir"]  = args.input_dir
    if args.output_dir: cfg["output_dir"] = args.output_dir
    if args.audio:      cfg["audio"]      = args.audio
    if args.key:        cfg["passphrase"] = args.key
    cfg.setdefault("log", args.log)

    # Validate required settings
    missing = [k for k in ("input_dir", "output_dir", "passphrase") if not cfg.get(k)]
    if missing:
        parser.error(
            f"Missing required setting(s): {', '.join(missing)}. "
            "Provide them via CLI flags or a --config JSON file."
        )

    os.makedirs(cfg["input_dir"],  exist_ok=True)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    logger  = _setup_logger(cfg["log"])
    handler = _ImageHandler(cfg, logger)

    observer = Observer()
    observer.schedule(handler, cfg["input_dir"], recursive=False)
    observer.start()

    _audit(logger, {
        "event":      "WATCH_STARTED",
        "input_dir":  cfg["input_dir"],
        "output_dir": cfg["output_dir"],
        "audio":      cfg.get("audio", "synthetic"),
        "log":        cfg["log"],
    })
    logger.info(f"Watching : {cfg['input_dir']}")
    logger.info(f"Output   : {cfg['output_dir']}")
    logger.info(f"Audit log: {cfg['log']}")
    logger.info("Press Ctrl+C to stop.\n")

    try:
        while True:
            handler.poll()
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        _audit(logger, {"event": "WATCH_STOPPED"})
        logger.info("Watcher stopped cleanly.")


if __name__ == "__main__":
    main()
