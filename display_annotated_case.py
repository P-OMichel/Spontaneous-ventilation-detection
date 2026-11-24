import json
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy import signal

from Annotation_App.annotations import (
    AnnotationStore,
    REGIME_CLASSES,
    ID_TO_REGIME,
)

REGIME_PLOT_COLORS = {
    0: ("#3498db", 0.20),  # Controlled
    1: ("#95a5a6", 0.20),  # Unknown
    2: ("#e74c3c", 0.22),  # Apnea
    3: ("#f39c12", 0.24),  # Transition to Spontaneous ventilation
    4: ("#2ecc71", 0.22),  # Efficient spontaneous ventilation
}


def merge_window_labels_to_spans(
    t_windows: np.ndarray, labels: np.ndarray, step_sec: float = 2.0
) -> list[tuple[float, float, int]]:
    """Converts per-window integer labels into contiguous (t0, t1, regime_id) spans."""
    if len(t_windows) == 0 or labels is None:
        return []
    half_step = 0.5 * step_sec
    spans = []
    cur_id = int(labels[0])
    t0 = float(t_windows[0]) - half_step

    for i in range(1, len(t_windows)):
        rid = int(labels[i])
        if rid != cur_id:
            t1 = 0.5 * (float(t_windows[i - 1]) + float(t_windows[i]))
            spans.append((t0, t1, cur_id))
            t0 = t1
            cur_id = rid

    spans.append((t0, float(t_windows[-1]) + half_step, cur_id))
    return spans


def plot_saved_annotated_sample(annotated_npz_path: str, show_uncropped: bool = False):
    """
    Plots a saved *_vent_annotated.npz file.

    Args:
        annotated_npz_path: Path to the saved *_vent_annotated.npz archive.
        show_uncropped: If False (default), plots the cropped ML-ready segment via
                        VentilationAnnotationStore.load_cropped_for_ml().
                        If True, plots the full saved window with crop boundaries drawn.
    """
    if not show_uncropped:
        sample = AnnotationStore.load_cropped_for_ml(annotated_npz_path)
        meta = sample["metadata"]
        t = sample["t"]
        awp = sample["awp"]
        co2 = sample["co2"]
        eeg = sample["eeg"]
        t_win = sample["t_windows"]
        regimes = sample["regime_labels"]
        spont_times = sample["spont_onset_times"]
        extub_times = sample["extubation_times"]
        crop_bounds = None
    else:
        with np.load(annotated_npz_path, allow_pickle=False) as data:
            keys = set(data.files)
            meta = json.loads(str(data["metadata_json"]))
            t = data["t_common"]
            awp = data["awp"]
            co2 = data["co2"]
            eeg = data["eeg"] if "eeg" in keys else None
            t_win = data["t_windows"]
            regimes = data["regime_labels"] if "regime_labels" in keys else None
            spont_times = data["spont_onset_times"] if "spont_onset_times" in keys else None
            extub_times = data["extubation_times"] if "extubation_times" in keys else None
            crop_bounds = (float(meta["crop_start_sec"]), float(meta["crop_end_sec"]))
            sample = {
                "sample_id": str(data["sample_id"]),
                "hypnotic_type": str(data["hypnotic_type"]),
            }

    fs = float(meta.get("sampling_rate_hz", 128.0))
    step_sec = float(meta.get("step_size_sec", 2.0))
    spans = merge_window_labels_to_spans(t_win, regimes, step_sec=step_sec)

    fig, axes = plt.subplots(
        4, 1, figsize=(15, 11), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.1, 1.5, 1.1]}
    )
    plt.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.07, hspace=0.18)

    mode_str = "Uncropped Archive" if show_uncropped else "Cropped ML-Ready Segment"
    fig.suptitle(
        f"Annotated Sample: {sample['sample_id']} | Hypnotic: {sample['hypnotic_type']} | "
        f"Duration: {t[-1] - t[0]:.1f}s ({mode_str})",
        fontsize=12,
        fontweight="bold",
    )

    # --- 1. EEG Spectrogram (if EEG track is non-zero) ---
    if eeg is not None and len(eeg) > int(fs * 3) and np.any(eeg != 0):
        nyq = 0.5 * fs
        sos = signal.butter(4, [0.5 / nyq, min(45.0, 0.45 * fs) / nyq], btype="band", output="sos")
        eeg_filt = signal.sosfiltfilt(sos, eeg)
        freqs, t_spec, Sxx = signal.spectrogram(
            eeg_filt, fs=fs, nperseg=int(fs * 2), noverlap=int(fs * 1.5)
        )
        f_mask = (freqs >= 0.5) & (freqs <= 35.0)
        Sxx_db = 10.0 * np.log10(Sxx[f_mask, :] + 1e-12)
        vmin, vmax = np.percentile(Sxx_db, [5, 98])
        axes[0].pcolormesh(
            t[0] + t_spec, freqs[f_mask], Sxx_db, shading="auto", cmap="turbo", vmin=vmin, vmax=vmax
        )
    axes[0].set_ylabel("EEG Freq (Hz)")
    axes[0].set_title("EEG Spectrogram (0.5–35 Hz)", loc="left", fontsize=9)

    # --- 2. Capnography (CO2) ---
    axes[1].plot(t, co2, color="#1f77b4", linewidth=0.8)
    axes[1].set_ylabel("CO2 (mmHg)")
    axes[1].set_ylim(-5, 65)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Capnography (CO2)", loc="left", fontsize=9)

    # --- 3. Airway Pressure (AWP) + Ground-Truth Regime Shading ---
    axes[2].plot(t, awp, color="#2c3e50", linewidth=0.75)
    axes[2].axhline(0, color="#7f8c8d", linestyle="--", linewidth=0.7)
    axes[2].set_ylabel("AWP (hPa)")
    axes[2].set_ylim(-10, 50)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Airway Pressure (AWP) with Ground-Truth Regime Intervals", loc="left", fontsize=9)

    present_ids = set()
    for seg_t0, seg_t1, rid in spans:
        color, alpha = REGIME_PLOT_COLORS.get(rid, ("#95a5a6", 0.2))
        axes[2].axvspan(seg_t0, seg_t1, color=color, alpha=alpha, lw=0)
        present_ids.add(rid)

    if present_ids:
        handles = [
            Patch(facecolor=REGIME_PLOT_COLORS[rid][0], alpha=0.45, label=ID_TO_REGIME[rid])
            for rid in sorted(present_ids)
        ]
        axes[2].legend(handles=handles, loc="upper left", ncol=len(handles), fontsize=8)

    # --- 4. Ground-Truth Regime Step Timeline ---
    if regimes is not None and len(t_win) > 0:
        axes[3].step(t_win, regimes, where="mid", color="#34495e", linewidth=1.2)
        for rid in sorted(present_ids):
            mask = regimes == rid
            axes[3].scatter(
                t_win[mask],
                regimes[mask],
                color=REGIME_PLOT_COLORS[rid][0],
                s=14,
                zorder=3,
            )
    short_labels = ["Controlled", "Unknown", "Apnea", "Transition Spont.", "Efficient Spont."]
    axes[3].set_yticks(range(len(REGIME_CLASSES)))
    axes[3].set_yticklabels(short_labels, fontsize=8)
    axes[3].set_ylim(-0.5, len(REGIME_CLASSES) - 0.5)
    axes[3].set_ylabel("Regime")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].grid(True, alpha=0.3)

    # --- Overlay Saved Transition Events & Optional Crop Lines ---
    for ax in axes[1:]:
        if spont_times is not None:
            for ts in spont_times:
                ax.axvline(ts, color="#d4ac0d", linestyle="--", linewidth=1.6, label="Spont. Onset")
        if extub_times is not None:
            for te in extub_times:
                ax.axvline(te, color="#8e44ad", linestyle="--", linewidth=1.6, label="Extubation")
        if crop_bounds is not None:
            ax.axvline(crop_bounds[0], color="#e74c3c", linestyle="-", linewidth=1.8)
            ax.axvline(crop_bounds[1], color="#e74c3c", linestyle="-", linewidth=1.8)

    axes[3].set_xlim(t[0], t[-1])
    plt.show()



plot_saved_annotated_sample(r'db_cases_128hz/vitaldb_case_1_vent_annotated.npz')