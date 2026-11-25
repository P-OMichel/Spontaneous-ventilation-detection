import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Canonical display names & colors (aligned to Synthetic UNet1D IDs 0..4)
CANONICAL_NAMES = {
    0: "Controlled",
    1: "Transition",
    2: "Efficient Spontaneous",
    3: "Apnea",
    4: "Unknown / Artifact",
}

REGIME_COLORS = {
    0: "#1f4e78",  # Controlled (Steel Blue)
    1: "#7f6000",  # Transition (Amber / Brown)
    2: "#1e5631",  # Efficient Spontaneous (Dark Green)
    3: "#661d1d",  # Apnea (Deep Red)
    4: "#4a4a4a",  # Unknown / Artifact (Charcoal Grey)
}

# Mapping from Real AnnotationStore IDs -> Synthetic Model IDs
# Real:  0=Controlled, 1=Unknown, 2=Apnea, 3=Transition, 4=Efficient Spont
# Model: 0=Controlled, 1=Transition, 2=Efficient Spont, 3=Apnea, 4=Unknown
REAL_TO_MODEL_ID = {
    0: 0,  # Controlled -> Controlled
    1: 4,  # Unknown -> Unknown / Artifact
    2: 3,  # Apnea -> Apnea
    3: 1,  # Transition -> Transition
    4: 2,  # Efficient Spontaneous -> Efficient Spontaneous
}


# ==============================================================================
# 1. 1D U-NET MODEL DEFINITION (Matches Saved Checkpoint)
# ==============================================================================
class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, base_filters=32, dropout=0.0):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv1D(in_channels, f, kernel_size=7, dropout=dropout)
        self.pool1 = nn.MaxPool1d(4)
        self.enc2 = DoubleConv1D(f, f * 2, kernel_size=7, dropout=dropout)
        self.pool2 = nn.MaxPool1d(4)
        self.enc3 = DoubleConv1D(f * 2, f * 4, kernel_size=5, dropout=dropout)
        self.pool3 = nn.MaxPool1d(4)
        self.enc4 = DoubleConv1D(f * 4, f * 8, kernel_size=5, dropout=dropout)
        self.pool4 = nn.MaxPool1d(4)

        self.bottleneck = DoubleConv1D(f * 8, f * 16, kernel_size=5, dropout=dropout)

        self.up4 = nn.ConvTranspose1d(f * 16, f * 8, kernel_size=4, stride=4)
        self.dec4 = DoubleConv1D(f * 16, f * 8, kernel_size=5, dropout=dropout)
        self.up3 = nn.ConvTranspose1d(f * 8, f * 4, kernel_size=4, stride=4)
        self.dec3 = DoubleConv1D(f * 8, f * 4, kernel_size=5, dropout=dropout)
        self.up2 = nn.ConvTranspose1d(f * 4, f * 2, kernel_size=4, stride=4)
        self.dec2 = DoubleConv1D(f * 4, f * 2, kernel_size=7, dropout=dropout)
        self.up1 = nn.ConvTranspose1d(f * 2, f, kernel_size=4, stride=4)
        self.dec1 = DoubleConv1D(f * 2, f, kernel_size=7, dropout=dropout)

        self.head = nn.Conv1d(f, num_classes, kernel_size=1)

    def _pad_and_cat(self, up_x, skip_x):
        diff = skip_x.size(-1) - up_x.size(-1)
        if diff > 0:
            up_x = F.pad(up_x, (0, diff))
        elif diff < 0:
            up_x = up_x[..., :skip_x.size(-1)]
        return torch.cat([skip_x, up_x], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(self._pad_and_cat(self.up4(b), e4))
        d3 = self.dec3(self._pad_and_cat(self.up3(d4), e3))
        d2 = self.dec2(self._pad_and_cat(self.up2(d3), e2))
        d1 = self.dec1(self._pad_and_cat(self.up1(d2), e1))
        return self.head(d1)


def load_trained_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["model_config"]
    model = UNet1D(
        in_channels=cfg["in_channels"],
        num_classes=cfg["num_classes"],
        base_filters=cfg["base_filters"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mean = np.array(ckpt["normalization"]["mean"], dtype=np.float32)
    std = np.array(ckpt["normalization"]["std"], dtype=np.float32)
    return model, mean, std, cfg["window_samples"], ckpt["fs"]


# ==============================================================================
# 2. REAL DATA LOADER & PREPROCESSOR (*_vent_annotated.npz)
# ==============================================================================
def _estimate_surrogate_cdyn(awp, fs, default_cdyn=55.0):
    """
    Approximates sample-and-hold Dynamic Compliance (mL/mbar) when real archive
    lacks a dedicated compliance channel, assuming nominal ~480 mL tidal volume
    during active positive-pressure cycles.
    """
    n = len(awp)
    cdyn = np.full(n, default_cdyn, dtype=np.float32)
    step = int(8 * fs)
    last_val = default_cdyn
    for s in range(0, n, step):
        e = min(n, s + step)
        dp = float(np.max(awp[s:e]) - np.min(awp[s:e]))
        if dp > 6.0:
            last_val = float(np.clip(480.0 / dp, 15.0, 90.0))
        cdyn[s:e] = last_val
    return cdyn


def _upsample_window_labels_to_samples(t_samples, t_windows, win_labels, step_sec=2.0):
    """Maps 2.0s window annotations onto the 128 Hz sample timeline and remaps IDs."""
    if win_labels is None or len(t_windows) == 0:
        return None
    y_samples = np.full(len(t_samples), fill_value=4, dtype=np.int64)  # Default Unknown
    half_step = 0.5 * step_sec
    for tw, raw_id in zip(t_windows, win_labels):
        mapped_id = REAL_TO_MODEL_ID.get(int(raw_id), 4)
        mask = (t_samples >= (tw - half_step)) & (t_samples < (tw + half_step))
        y_samples[mask] = mapped_id
    return y_samples


def load_real_annotated_case(annotated_npz_path, use_cropped=True, default_cdyn=55.0):
    """
    Loads a real *_vent_annotated.npz file (using AnnotationStore if available,
    otherwise reading the .npz archive directly) and returns model-ready (3, L) signals.
    """
    try:
        from Annotation_App.annotations import AnnotationStore
        if use_cropped:
            sample = AnnotationStore.load_cropped_for_ml(annotated_npz_path)
            meta = sample["metadata"]
            t = sample["t"]
            awp = sample["awp"]
            co2 = sample["co2"]
            t_win = sample["t_windows"]
            regimes = sample["regime_labels"]
            case_label = f"{sample.get('sample_id', os.path.basename(annotated_npz_path))} (Cropped)"
        else:
            raise ImportError("Fall through to uncropped reader")
    except Exception:
        with np.load(annotated_npz_path, allow_pickle=False) as data:
            keys = set(data.files)
            meta = json.loads(str(data["metadata_json"]))
            t_full = data["t_common"]
            awp_full = data["awp"]
            co2_full = data["co2"]
            t_win = data["t_windows"] if "t_windows" in keys else np.array([])
            regimes = data["regime_labels"] if "regime_labels" in keys else None
            sample_id = str(data["sample_id"]) if "sample_id" in keys else os.path.basename(annotated_npz_path)

            if use_cropped and "crop_start_sec" in meta and "crop_end_sec" in meta:
                c0, c1 = float(meta["crop_start_sec"]), float(meta["crop_end_sec"])
                mask = (t_full >= c0) & (t_full <= c1)
                t, awp, co2 = t_full[mask], awp_full[mask], co2_full[mask]
                if len(t_win) > 0 and regimes is not None:
                    w_mask = (t_win >= c0) & (t_win <= c1)
                    t_win, regimes = t_win[w_mask], regimes[w_mask]
                case_label = f"{sample_id} (Cropped)"
            else:
                t, awp, co2 = t_full, awp_full, co2_full
                case_label = f"{sample_id} (Uncropped)"

    fs = float(meta.get("sampling_rate_hz", 128.0))
    step_sec = float(meta.get("step_size_sec", 2.0))

    # Clean any NaNs in real clinical sensors
    co2 = np.nan_to_num(co2, nan=0.0).astype(np.float32)
    awp = np.nan_to_num(awp, nan=0.0).astype(np.float32)
    cdyn = _estimate_surrogate_cdyn(awp, fs=fs, default_cdyn=default_cdyn)

    signals_3xL = np.stack([co2, awp, cdyn], axis=0)
    y_true_128hz = _upsample_window_labels_to_samples(t, t_win, regimes, step_sec=step_sec)

    return t, signals_3xL, y_true_128hz, fs, case_label


# ==============================================================================
# 3. SLIDING-WINDOW INFERENCE & UNIFIED COMPARISON PLOTTING
# ==============================================================================
@torch.no_grad()
def predict_single_recording(model, signals_3xL, mean, std, window_samples, device, stride_ratio=0.25):
    """
    Runs Hann-weighted sliding-window inference on any recording of shape (3, L).
    Handles recordings shorter or longer than window_samples cleanly.
    """
    _, total_len = signals_3xL.shape
    num_classes = model.head.out_channels
    norm_sig = (signals_3xL - mean[:, None]) / (std[:, None] + 1e-6)

    # If real clip is shorter than 60s window, pad right and slice back
    if total_len < window_samples:
        pad_len = window_samples - total_len
        padded = np.pad(norm_sig, ((0, 0), (0, pad_len)), mode="edge")
        xb = torch.from_numpy(padded).unsqueeze(0).float().to(device)
        probs = torch.softmax(model(xb), dim=1).squeeze(0).cpu().numpy()[:, :total_len]
        return np.argmax(probs, axis=0), probs

    stride = max(1, int(window_samples * stride_ratio))
    prob_accum = np.zeros((num_classes, total_len), dtype=np.float32)
    weight_accum = np.zeros((1, total_len), dtype=np.float32)
    win_weight = (np.hanning(window_samples).astype(np.float32) + 0.1)[None, :]

    starts = list(range(0, total_len - window_samples + 1, stride))
    if starts[-1] + window_samples < total_len:
        starts.append(total_len - window_samples)

    for s in starts:
        e = s + window_samples
        xb = torch.from_numpy(norm_sig[:, s:e]).unsqueeze(0).float().to(device)
        probs = torch.softmax(model(xb), dim=1).squeeze(0).cpu().numpy()
        prob_accum[:, s:e] += probs * win_weight
        weight_accum[:, s:e] += win_weight

    avg_probs = prob_accum / np.maximum(weight_accum, 1e-6)
    pred_labels = np.argmax(avg_probs, axis=0)
    return pred_labels, avg_probs


def _shade_regimes(ax, t, labels, alpha=0.45):
    change_idx = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(labels) - 1]))
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color=REGIME_COLORS[int(labels[s])], alpha=alpha, zorder=1)
        if s > 0:
            ax.axvline(t[s], color="#aaaaaa", linestyle=":", linewidth=0.8, alpha=0.7)


def plot_inference_comparison(t, co2, awp, cdyn, y_true, y_pred, case_title, fs=128):
    has_truth = y_true is not None
    if has_truth:
        acc = (y_true == y_pred).mean() * 100.0
        present_classes = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
        macro_f1 = f1_score(y_true, y_pred, labels=present_classes, average="macro", zero_division=0)
        header_metrics = f" | Accuracy: {acc:.2f}% | Macro-F1: {macro_f1:.4f}"
    else:
        header_metrics = " | (No Ground Truth Labels)"

    plt.style.use("dark_background")
    n_rows = 4 if has_truth else 3
    ratios = [1.1, 1.4, 1.4, 0.6] if has_truth else [1.1, 1.5, 0.6]
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(16, 10 if has_truth else 8), sharex=True,
        gridspec_kw={"height_ratios": ratios, "hspace": 0.22}
    )
    fig.patch.set_facecolor("#121212")

    # 1. CO2
    axes[0].set_facecolor("#181818")
    axes[0].plot(t, co2, color="#29b6f6", linewidth=1.15)
    axes[0].set_ylabel("CO2 (mmHg)", fontsize=9.5)
    axes[0].set_title(f"{case_title} — 1D U-Net Inference{header_metrics}", fontsize=11.5, color="#eeeeee", pad=8)

    if has_truth:
        # 2. AWP with Ground-Truth Shading
        axes[1].set_facecolor("#181818")
        axes[1].axhline(0, color="#888888", linestyle="--", linewidth=0.8)
        axes[1].plot(t, awp, color="#f5f5f5", linewidth=0.95, zorder=2)
        _shade_regimes(axes[1], t, y_true, alpha=0.45)
        axes[1].set_ylabel("AWP (hPa)\n[TRUE]", fontsize=9.5, color="#81c784")
        pred_ax = axes[2]
        cdyn_ax = axes[3]
    else:
        pred_ax = axes[1]
        cdyn_ax = axes[2]

    # AWP with U-Net Predicted Shading + Error Highlights
    pred_ax.set_facecolor("#181818")
    pred_ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    pred_ax.plot(t, awp, color="#f5f5f5", linewidth=0.95, zorder=2)
    _shade_regimes(pred_ax, t, y_pred, alpha=0.45)
    pred_ax.set_ylabel("AWP (hPa)\n[PREDICTED]", fontsize=9.5, color="#ffd54f")

    if has_truth:
        mismatch = y_true != y_pred
        if np.any(mismatch):
            ymin = pred_ax.get_ylim()[0]
            pred_ax.scatter(t[mismatch], np.full(mismatch.sum(), ymin + 0.8), color="#ff4081", s=4, zorder=4)

    # Dynamic Compliance (or Surrogate Cdyn)
    cdyn_ax.set_facecolor("#181818")
    cdyn_ax.plot(t, cdyn, color="#4db6ac", linewidth=1.3)
    dot_step = max(1, int(8 * fs))
    cdyn_ax.scatter(t[::dot_step], cdyn[::dot_step], color="#80cbc4", s=20)
    cdyn_ax.set_ylabel("Cdyn", fontsize=9)
    cdyn_ax.set_xlabel("Time (seconds)", fontsize=10)

    for ax in axes:
        ax.grid(True, color="#2a2a2a", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color=REGIME_COLORS[k], alpha=0.7, label=CANONICAL_NAMES[k]) for k in sorted(CANONICAL_NAMES)]
    axes[1].legend(handles=patches, loc="upper right", ncol=5, frameon=True, facecolor="#1e1e1e", fontsize=8.5)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. MAIN ENTRYPOINT: SWITCH BETWEEN "synthetic" AND "real"
# ==============================================================================
if __name__ == "__main__":
    CHECKPOINT_PATH = "unet_baseline/best_unet1d.pt"

    # Choose data source: "synthetic" or "real"
    DATA_SOURCE = "real"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, win_samples, train_fs = load_trained_model(CHECKPOINT_PATH, device)

    if DATA_SOURCE == "synthetic":
        TEST_NPZ_PATH = "anesthesia_synth_dataset/test.npz"
        test_data = np.load(TEST_NPZ_PATH)
        X_test, y_test = test_data["X"], test_data["y"]

        for case_idx in [0, 1, 2]:
            signals = X_test[case_idx]
            y_true = y_test[case_idx]
            y_pred, _ = predict_single_recording(model, signals, mean, std, win_samples, device)

            t = np.arange(len(y_true)) / train_fs
            co2, awp, cdyn = signals
            plot_inference_comparison(
                t, co2, awp, cdyn, y_true, y_pred,
                case_title=f"Synthetic Test Case #{case_idx}", fs=train_fs
            )

    elif DATA_SOURCE == "real":
        REAL_FILES = [
            r"db_cases_128hz/vitaldb_case_1_vent_annotated.npz",
            # Add more real case paths here, e.g.:
            # r"db_cases_128hz/vitaldb_case_2_vent_annotated.npz",
        ]
        USE_CROPPED_SEGMENT = True  # Matches AnnotationStore.load_cropped_for_ml()

        for file_path in REAL_FILES:
            t, signals, y_true, fs, case_label = load_real_annotated_case(
                file_path, use_cropped=USE_CROPPED_SEGMENT, default_cdyn=float(mean[2])
            )
            y_pred, _ = predict_single_recording(model, signals, mean, std, win_samples, device)

            if y_true is not None:
                present = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
                print(f"\n=== Classification Report: {case_label} ===")
                print(classification_report(
                    y_true, y_pred,
                    labels=present,
                    target_names=[CANONICAL_NAMES[i] for i in present],
                    digits=4,
                    zero_division=0
                ))

            co2, awp, cdyn = signals
            plot_inference_comparison(
                t, co2, awp, cdyn, y_true, y_pred,
                case_title=f"Real VitalDB: {case_label}", fs=fs
            )