"""
app_gui.py - CryptoStego PySide6 Desktop Application
===================================================
Native Windows GUI for secure digital image transmission.
All crypto / stego logic is delegated to the backend modules —
no encryption code exists here.

Launch:
    python app_gui.py

Build standalone .exe:
    pyinstaller cryptostego.spec --clean
"""
import hashlib
import os
import secrets
import shutil
import sys
import tempfile

import cv2
import numpy as np
from scipy.io import wavfile

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QSizePolicy, QTabWidget, QVBoxLayout, QWidget, QGridLayout,
)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    _MEDIA_OK = True
except ImportError:
    _MEDIA_OK = False

# ── Backend (same functions as the CLI) ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_stego
import image_io
import metrics
import pipeline as _pipe
from pipeline import receiver as _recv, sender as _send, _payload_size, _synth_audio


# ═════════════════════════════════════════════════════════════════════════════
# WORKER THREADS  (all heavy work off the main / UI thread)
# ═════════════════════════════════════════════════════════════════════════════

class ProtectWorker(QThread):
    progress = Signal(int, str)   # (percent, message)
    finished = Signal(object)     # result dict
    error    = Signal(str)        # human-readable error

    def __init__(self, image_paths, n_lsbs, audio_path=None):
        super().__init__()
        self.image_paths = image_paths
        self.n_lsbs      = n_lsbs
        self.audio_path  = audio_path

    def run(self):
        try:
            self.progress.emit(5, "Loading images...")
            images, metas = [], []
            for p in self.image_paths:
                arr, meta = image_io.load(p)
                images.append(arr)
                metas.append(meta)

            # Auto-generate encryption key
            self.progress.emit(15, "Generating encryption key...")
            passphrase = secrets.token_hex(16)

            needed_bits = _payload_size(images) * 8

            if self.audio_path:
                # User-supplied cover audio
                self.progress.emit(20, "Loading cover audio...")
                sr, cover = wavfile.read(self.audio_path)
                if cover.ndim > 1:
                    cover = cover[:, 0]
                cover    = cover.astype(np.int16)
                cap_bits = audio_stego.capacity_bits(cover, self.n_lsbs)
            else:
                # Auto-generate synthetic cover audio sized exactly for this payload
                self.progress.emit(20, "Measuring carrier capacity...")
                sr = 44100  # CD-quality: ~2.75× more capacity than 16 kHz

                # Probe a 5-second sample to measure actual carrier bits/second
                probe     = _synth_audio(5.0, sr)
                probe_cap = audio_stego.capacity_bits(probe, self.n_lsbs)
                bits_per_sec = probe_cap / 5.0  # exact measured rate

                # Calculate precise duration + 3% safety margin
                exact_dur = max((needed_bits / bits_per_sec) * 1.03, 5.0)
                self.progress.emit(25, f"Generating {exact_dur:.1f}s cover audio...")
                cover    = _synth_audio(exact_dur, sr)
                cap_bits = audio_stego.capacity_bits(cover, self.n_lsbs)
                # Fallback if probe underestimated (rare)
                if needed_bits > cap_bits:
                    exact_dur *= (needed_bits / cap_bits) * 1.05
                    cover     = _synth_audio(exact_dur, sr)
                    cap_bits  = audio_stego.capacity_bits(cover, self.n_lsbs)

            self.progress.emit(40, "Encrypting images...")
            ciphers, stego, km_list = _send(
                images, cover, passphrase, n_lsbs=self.n_lsbs
            )

            self.progress.emit(90, "Writing stego file...")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wavfile.write(tmp.name, sr, stego)
            tmp.close()

            snr_db  = metrics.snr(cover, stego)
            cap_pct = needed_bits / cap_bits * 100

            self.progress.emit(100, "Complete.")
            self.finished.emit({
                "images":     images,
                "metas":      metas,
                "ciphers":    ciphers,
                "stego":      stego,
                "cover":      cover,
                "sr":         sr,
                "km_list":    km_list,
                "stego_tmp":  tmp.name,
                "n_lsbs":     self.n_lsbs,
                "snr_db":     snr_db,
                "cap_pct":    cap_pct,
                "passphrase": passphrase,
                "audio_src":  os.path.basename(self.audio_path) if self.audio_path
                              else f"auto-generated ({len(cover)/sr:.1f}s synthetic)",
            })
        except Exception as exc:
            self.error.emit(str(exc))


class RecoverWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(object)
    error    = Signal(str)

    def __init__(self, stego_path, passphrase):
        super().__init__()
        self.stego_path = stego_path
        self.passphrase = passphrase

    def run(self):
        try:
            self.progress.emit(10, "Loading stego audio...")
            sr, stego = wavfile.read(self.stego_path)
            if stego.ndim > 1:
                stego = stego[:, 0]
            stego = stego.astype(np.int16)

            self.progress.emit(30, "Extracting encrypted payload...")
            payload = audio_stego.extract(stego)
            pairs   = _pipe._deserialize_ciphers(payload)

            self.progress.emit(55, "Decrypting images...")
            recovered = _recv(stego, self.passphrase)

            self.progress.emit(80, "Verifying SHA-256 integrity...")
            integrity = [
                hashlib.sha256(np.ascontiguousarray(rec).tobytes()).digest() == digest
                for rec, (_, digest) in zip(recovered, pairs)
            ]

            self.progress.emit(100, "Complete.")
            self.finished.emit({"recovered": recovered, "integrity": integrity})
        except Exception as exc:
            msg = str(exc)
            if "magic" in msg.lower():
                msg = (
                    "This file does not appear to be a CryptoStego stego WAV, "
                    "or it may be corrupted.\n\nDetails: " + msg
                )
            self.error.emit(msg)


class EvalWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(str)   # output_dir
    error    = Signal(str)

    def __init__(self, result_dict, outdir):
        super().__init__()
        self._r     = result_dict
        self.outdir = outdir

    def run(self):
        try:
            import evaluation as ev
            self.progress.emit(10, "Running evaluation suite...")
            ev.evaluate(
                images     = self._r["images"],
                ciphers    = self._r["ciphers"],
                recovered  = _recv(self._r["stego"], self._r.get("passphrase","?")),
                km_list    = self._r["km_list"],
                stego      = self._r["stego"],
                cover      = self._r["cover"],
                passphrase = self._r.get("passphrase", ""),
                output_dir = self.outdir,
                n_lsbs     = self._r.get("n_lsbs", 1),
            )
            self.progress.emit(100, "Evaluation complete.")
            self.finished.emit(self.outdir)
        except Exception as exc:
            self.error.emit(str(exc))


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _to_pixmap(arr: np.ndarray, max_px: int = 140) -> QPixmap:
    """Convert any image array to a QPixmap thumbnail."""
    disp = image_io.to_uint8_display(arr)
    h, w = disp.shape[:2]
    if max(h, w) > max_px:
        scale  = max_px / max(h, w)
        disp   = cv2.resize(disp, (max(1, int(w * scale)), max(1, int(h * scale))))
        h, w   = disp.shape[:2]
    bpl = w * 3
    qi  = QImage(disp.tobytes(), w, h, bpl, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi)


def _styled_label(text: str, ok: bool) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "color:#155724; font-weight:bold;" if ok
        else "color:#721c24; font-weight:bold;"
    )
    return lbl


# ═════════════════════════════════════════════════════════════════════════════
# PROTECT TAB
# ═════════════════════════════════════════════════════════════════════════════

class ProtectTab(QWidget):
    def __init__(self):
        super().__init__()
        self._image_paths = []
        self._audio_path  = None
        self._result      = None
        self._worker      = None
        self._player      = None
        self._audio_out   = None
        self._stego_tmp   = None
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Step 1: Images ───────────────────────────────────────────────────
        g1 = QGroupBox("Step 1 — Digital Images")
        l1 = QVBoxLayout(g1)
        row1 = QHBoxLayout()
        self._btn_add   = QPushButton("Add Images…")
        self._btn_clear = QPushButton("Clear")
        self._lbl_files = QLabel("No images selected")
        row1.addWidget(self._btn_add)
        row1.addWidget(self._btn_clear)
        row1.addWidget(self._lbl_files)
        row1.addStretch()
        l1.addLayout(row1)
        self._btn_add.clicked.connect(self._add_images)
        self._btn_clear.clicked.connect(self._clear_images)
        root.addWidget(g1)

        # ── Step 2: Cover Audio (optional) ───────────────────────────────────
        g_audio = QGroupBox("Step 2 — Cover Audio (optional)")
        la = QVBoxLayout(g_audio)
        rowa = QHBoxLayout()
        self._btn_load_audio  = QPushButton("Load Cover Audio (WAV)…")
        self._btn_clear_audio = QPushButton("Use Auto-Generated Audio")
        self._lbl_audio = QLabel("Auto-generated — sized to fit the payload")
        rowa.addWidget(self._btn_load_audio)
        rowa.addWidget(self._btn_clear_audio)
        rowa.addWidget(self._lbl_audio, 1)
        la.addLayout(rowa)
        self._btn_load_audio.clicked.connect(self._load_audio)
        self._btn_clear_audio.clicked.connect(self._clear_audio)
        root.addWidget(g_audio)

        # ── Step 3: LSBs ─────────────────────────────────────────────────────
        g2 = QGroupBox("Step 3 — Embedding Depth")
        l2 = QHBoxLayout(g2)
        self._lsb_group = QButtonGroup()
        rb1 = QRadioButton("1 LSB — higher quality (~90 dB SNR)")
        rb2 = QRadioButton("2 LSBs — double capacity (~84 dB SNR)")
        rb1.setChecked(True)
        self._lsb_group.addButton(rb1, 1)
        self._lsb_group.addButton(rb2, 2)
        l2.addWidget(rb1)
        l2.addWidget(rb2)
        l2.addStretch()
        root.addWidget(g2)

        # ── Run button + progress ─────────────────────────────────────────────
        self._btn_run  = QPushButton("Protect Images")
        self._btn_run.setMinimumHeight(40)
        font = self._btn_run.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self._btn_run.setFont(font)
        self._progress   = QProgressBar()
        self._progress.setVisible(False)
        self._lbl_status = QLabel("")
        root.addWidget(self._btn_run)
        root.addWidget(self._progress)
        root.addWidget(self._lbl_status)
        self._btn_run.clicked.connect(self._run)

        # ── Results ───────────────────────────────────────────────────────────
        self._grp_result = QGroupBox("Results")
        self._grp_result.setVisible(False)
        rl = QVBoxLayout(self._grp_result)

        # ── Generated key display ─────────────────────────────────────────────
        key_grp = QGroupBox("Your Encryption Key  —  SAVE THIS. You need it to recover your images.")
        key_grp.setStyleSheet("QGroupBox { color: #c0392b; font-weight: bold; }")
        key_row = QHBoxLayout(key_grp)
        self._lbl_key = QLineEdit()
        self._lbl_key.setReadOnly(True)
        self._lbl_key.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 13px; "
            "font-weight: bold; color: #2c3e50; background: #ecf0f1;"
        )
        self._btn_copy = QPushButton("Copy Key")
        self._btn_copy.setMinimumWidth(90)
        self._btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(self._lbl_key.text())
        )
        key_row.addWidget(self._lbl_key, 1)
        key_row.addWidget(self._btn_copy)
        rl.addWidget(key_grp)

        # thumbnail scroll area
        self._scroll     = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(200)
        self._thumb_host = QWidget()
        self._thumb_grid = QGridLayout(self._thumb_host)
        self._scroll.setWidget(self._thumb_host)
        rl.addWidget(self._scroll)

        # audio player row
        ar = QHBoxLayout()
        self._btn_play    = QPushButton("Play Stego Audio")
        self._btn_stop    = QPushButton("Stop")
        self._lbl_metrics = QLabel("")
        ar.addWidget(self._btn_play)
        ar.addWidget(self._btn_stop)
        ar.addWidget(self._lbl_metrics)
        ar.addStretch()
        rl.addLayout(ar)

        # save / eval row
        sr_row = QHBoxLayout()
        self._btn_save = QPushButton("Save Protected File (WAV)…")
        self._btn_eval = QPushButton("Run Evaluation Suite…")
        sr_row.addWidget(self._btn_save)
        sr_row.addWidget(self._btn_eval)
        rl.addLayout(sr_row)

        self._btn_play.clicked.connect(self._play_stego)
        self._btn_stop.clicked.connect(self._stop_audio)
        self._btn_save.clicked.connect(self._save_stego)
        self._btn_eval.clicked.connect(self._run_eval)

        root.addWidget(self._grp_result)
        root.addStretch()

    # ── slots ────────────────────────────────────────────────────────────────

    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Digital Images", "",
            "Digital Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.dcm);;"
            "All Files (*)"
        )
        if paths:
            self._image_paths.extend(paths)
            self._lbl_files.setText(
                f"{len(self._image_paths)} image(s) selected"
            )

    def _clear_images(self):
        self._image_paths.clear()
        self._lbl_files.setText("No images selected")

    def _load_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Cover Audio", "",
            "WAV Files (*.wav);;All Files (*)"
        )
        if path:
            self._audio_path = path
            self._lbl_audio.setText(os.path.basename(path))

    def _clear_audio(self):
        self._audio_path = None
        self._lbl_audio.setText("Auto-generated — sized to fit the payload")

    def _run(self):
        if not self._image_paths:
            QMessageBox.warning(self, "CryptoStego", "Please add at least one image.")
            return

        n_lsbs = self._lsb_group.checkedId()
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._grp_result.setVisible(False)

        self._worker = ProtectWorker(self._image_paths, n_lsbs, self._audio_path)
        self._worker.progress.connect(lambda p, m: (
            self._progress.setValue(p),
            self._lbl_status.setText(m),
        ))
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, res):
        self._result    = res
        self._stego_tmp = res["stego_tmp"]
        self._btn_run.setEnabled(True)

        # Show generated key
        self._lbl_key.setText(res["passphrase"])

        # clear old thumbnails
        while self._thumb_grid.count():
            w = self._thumb_grid.itemAt(0).widget()
            self._thumb_grid.removeWidget(w)
            w.deleteLater()

        # header
        for col, txt in enumerate(["Original", "Cipher (noise)", "Metrics"]):
            self._thumb_grid.addWidget(QLabel(f"<b>{txt}</b>"), 0, col)

        for i, (img, cipher) in enumerate(zip(res["images"], res["ciphers"])):
            pm_o = _to_pixmap(img)
            pm_c = _to_pixmap(cipher)
            lbl_o = QLabel(); lbl_o.setPixmap(pm_o)
            lbl_c = QLabel(); lbl_c.setPixmap(pm_c)

            npcr_v = metrics.npcr(img, cipher)
            ent_v  = metrics.shannon_entropy(cipher)
            cc     = metrics.correlation_coefficients(cipher)
            info = (
                f"<b>Image {i:02d}</b><br>"
                f"Size: {img.shape}<br>"
                f"Dtype: {img.dtype}<br>"
                f"NPCR: {npcr_v:.2f}%<br>"
                f"Entropy: {ent_v:.4f} bits<br>"
                f"Corr H/V/D: {cc['horizontal']:.3f} / "
                f"{cc['vertical']:.3f} / {cc['diagonal']:.3f}"
            )
            lbl_i = QLabel(info)
            lbl_i.setTextFormat(Qt.TextFormat.RichText)

            self._thumb_grid.addWidget(lbl_o, i+1, 0)
            self._thumb_grid.addWidget(lbl_c, i+1, 1)
            self._thumb_grid.addWidget(lbl_i, i+1, 2)

        self._lbl_metrics.setText(
            f"Images: {res['n_lsbs']} LSB | "
            f"Capacity used: {res['cap_pct']:.1f}% | "
            f"Stego SNR: {res['snr_db']:.1f} dB | "
            f"Cover: {res['audio_src']}"
        )
        self._grp_result.setVisible(True)
        self._lbl_status.setText("Protection complete.")

    def _on_error(self, msg):
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        QMessageBox.critical(self, "CryptoStego — Protection Error", msg)

    def _play_stego(self):
        if not self._stego_tmp:
            return
        if not _MEDIA_OK:
            # Fallback: system default player
            import subprocess
            subprocess.Popen(["cmd", "/c", "start", "", self._stego_tmp],
                             shell=False)
            return
        if self._player is None:
            self._player    = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_out)
        self._player.setSource(QUrl.fromLocalFile(
            os.path.abspath(self._stego_tmp)
        ))
        self._player.play()

    def _stop_audio(self):
        if self._player:
            self._player.stop()

    def _save_stego(self):
        if not self._stego_tmp:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Protected File", "cryptostego_output.wav",
            "WAV Files (*.wav)"
        )
        if path:
            shutil.copy(self._stego_tmp, path)
            QMessageBox.information(self, "CryptoStego",
                                    f"Saved to:\n{path}")

    def _run_eval(self):
        if not self._result:
            return
        outdir = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Evaluation"
        )
        if not outdir:
            return
        self._eval_worker = EvalWorker(self._result, outdir)
        self._eval_worker.progress.connect(lambda p, m: self._lbl_status.setText(m))
        self._eval_worker.finished.connect(lambda d: QMessageBox.information(
            self, "CryptoStego",
            f"Evaluation complete.\nOutputs saved to:\n{d}"
        ))
        self._eval_worker.error.connect(lambda e: QMessageBox.critical(
            self, "CryptoStego — Evaluation Error", e
        ))
        self._eval_worker.start()


# ═════════════════════════════════════════════════════════════════════════════
# RECOVER TAB
# ═════════════════════════════════════════════════════════════════════════════

class RecoverTab(QWidget):
    def __init__(self):
        super().__init__()
        self._stego_path = None
        self._result     = None
        self._worker     = None
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Step 1: Stego WAV ─────────────────────────────────────────────────
        g1 = QGroupBox("Step 1 — Stego Audio File")
        l1 = QHBoxLayout(g1)
        self._btn_open = QPushButton("Open Stego WAV…")
        self._lbl_file = QLabel("No file selected")
        l1.addWidget(self._btn_open)
        l1.addWidget(self._lbl_file, 1)
        self._btn_open.clicked.connect(self._open_stego)
        root.addWidget(g1)

        # ── Step 2: Key ──────────────────────────────────────────────────────
        g2 = QGroupBox("Step 2 — Decryption Key")
        l2 = QHBoxLayout(g2)
        self._key_edit = QLineEdit()
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText("Enter the passphrase from the sender…")
        l2.addWidget(QLabel("Passphrase:"))
        l2.addWidget(self._key_edit, 1)
        root.addWidget(g2)

        # ── Run ───────────────────────────────────────────────────────────────
        self._btn_run    = QPushButton("Extract & Decrypt")
        self._btn_run.setMinimumHeight(36)
        font = self._btn_run.font(); font.setBold(True)
        self._btn_run.setFont(font)
        self._progress   = QProgressBar()
        self._progress.setVisible(False)
        self._lbl_status = QLabel("")
        root.addWidget(self._btn_run)
        root.addWidget(self._progress)
        root.addWidget(self._lbl_status)
        self._btn_run.clicked.connect(self._run)

        # ── Results ───────────────────────────────────────────────────────────
        self._grp_result = QGroupBox("Recovered Images")
        self._grp_result.setVisible(False)
        rl = QVBoxLayout(self._grp_result)

        self._scroll     = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(220)
        self._thumb_host = QWidget()
        self._thumb_grid = QGridLayout(self._thumb_host)
        self._scroll.setWidget(self._thumb_host)
        rl.addWidget(self._scroll)

        self._btn_save = QPushButton("Save Recovered Images…")
        self._btn_save.clicked.connect(self._save_recovered)
        rl.addWidget(self._btn_save)

        root.addWidget(self._grp_result)
        root.addStretch()

    # ── slots ────────────────────────────────────────────────────────────────

    def _open_stego(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Stego WAV", "",
            "WAV Files (*.wav);;All Files (*)"
        )
        if path:
            self._stego_path = path
            self._lbl_file.setText(os.path.basename(path))

    def _run(self):
        if not self._stego_path:
            QMessageBox.warning(self, "CryptoStego", "Please open a stego WAV file.")
            return
        key = self._key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "CryptoStego", "Please enter the decryption key.")
            return

        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._grp_result.setVisible(False)

        self._worker = RecoverWorker(self._stego_path, key)
        self._worker.progress.connect(lambda p, m: (
            self._progress.setValue(p),
            self._lbl_status.setText(m),
        ))
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, res):
        self._result = res
        self._btn_run.setEnabled(True)
        self._lbl_status.setText(
            f"Recovery complete. "
            f"{'All integrity checks passed.' if all(res['integrity']) else 'WARNING: Some integrity checks FAILED.'}"
        )

        while self._thumb_grid.count():
            w = self._thumb_grid.itemAt(0).widget()
            self._thumb_grid.removeWidget(w)
            w.deleteLater()

        for col, txt in enumerate(["Recovered Image", "Integrity", "Info"]):
            self._thumb_grid.addWidget(QLabel(f"<b>{txt}</b>"), 0, col)

        for i, (img, ok) in enumerate(zip(res["recovered"], res["integrity"])):
            pm  = _to_pixmap(img)
            lbl = QLabel(); lbl.setPixmap(pm)

            tag = "SHA-256: PASS | MSE = 0" if ok else "SHA-256: FAIL (wrong key?)"
            lbl_int = _styled_label(tag, ok)

            lbl_info = QLabel(
                f"Shape: {img.shape}<br>Dtype: {img.dtype}"
            )
            lbl_info.setTextFormat(Qt.TextFormat.RichText)

            self._thumb_grid.addWidget(lbl,      i+1, 0)
            self._thumb_grid.addWidget(lbl_int,  i+1, 1)
            self._thumb_grid.addWidget(lbl_info, i+1, 2)

        self._grp_result.setVisible(True)

    def _on_error(self, msg):
        self._btn_run.setEnabled(True)
        self._progress.setVisible(False)
        QMessageBox.critical(self, "CryptoStego — Recovery Error", msg)

    def _save_recovered(self):
        if not self._result:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder"
        )
        if not folder:
            return
        saved = []
        for i, img in enumerate(self._result["recovered"]):
            path = os.path.join(folder, f"recovered_{i:02d}.png")
            try:
                saved_path = image_io.save(path, img)
                saved.append(saved_path)
            except Exception as e:
                QMessageBox.warning(self, "CryptoStego",
                                    f"Could not save image {i}: {e}")
        if saved:
            QMessageBox.information(
                self, "CryptoStego",
                f"Saved {len(saved)} image(s) to:\n{folder}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CryptoStego — General-Purpose Secure Image Transmission")
        self.setMinimumSize(960, 720)
        self.resize(1100, 800)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.addTab(ProtectTab(), "  Protect Images  ")
        tabs.addTab(RecoverTab(), "  Recover Images  ")
        self.setCentralWidget(tabs)

        # status bar
        self.statusBar().showMessage(
            "CryptoStego ready  —  no data is written to disk until you click Save"
        )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CryptoStego")
    app.setOrganizationName("CryptoStego Project")

    # default font
    f = QFont("Segoe UI", 9)
    app.setFont(f)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
