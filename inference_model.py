import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REGIME_COLORS = {
    0: "#1f4e78",  # Controlled (Steel Blue)
    1: "#7f6000",  # Transition (Amber / Brown)
    2: "#1e5631",  # Efficient Spontaneous (Dark Green)
    3: "#661d1d",  # Apnea (Deep Red)
    4: "#4a4a4a",  # Unknown / Artifact (Charcoal Grey)
}


# ==============================================================================
# 1. MODEL DEFINITION (Must match saved checkpoint)
# ==============================================================================
class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=0.1):
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


# ==============================================================================
# 2. INFERENCE & COMPARISON PLOTTING
# ==============================================================================
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
    regime_names = {int(k): v for k, v in ckpt["regime_names"].items()}
    return model, mean, std, cfg["window_samples"], ckpt["fs"], regime_names


@torch.no_grad()
def predict_single_recording(model, signals_3xL, mean, std, window_samples, device, stride_ratio=0.25):
    """
    Runs sliding-window inference on a single recording of shape (3, L).
    Returns:
      - pred_labels: shape (L,) integer class predictions
      - pred_probs:  shape (num_classes, L) class probabilities
    """
    _, total_len = signals_3xL.shape
    num_classes = model.head.out_channels
    stride = max(1, int(window_samples * stride_ratio))

    prob_accum = np.zeros((num_classes, total_len), dtype=np.float32)
    weight_accum = np.zeros((1, total_len), dtype=np.float32)
    win_weight = (np.hanning(window_samples).astype(np.float32) + 0.1)[None, :]

    starts = list(range(0, total_len - window_samples + 1, stride))
    if starts[-1] + window_samples < total_len:
        starts.append(total_len - window_samples)

    norm_sig = (signals_3xL - mean[:, None]) / (std[:, None] + 1e-6)

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


def plot_inference_comparison(t, co2, awp, cdyn, y_true, y_pred, regime_names, case_idx, fs=128):
    acc = (y_true == y_pred).mean() * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        4, 1, figsize=(16, 10), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.4, 1.4, 0.6], "hspace": 0.22}
    )
    fig.patch.set_facecolor("#121212")

    # 1. CO2
    axes[0].set_facecolor("#181818")
    axes[0].plot(t, co2, color="#29b6f6", linewidth=1.15)
    axes[0].set_ylabel("CO2 (mmHg)", fontsize=9.5)
    axes[0].set_title(
        f"Test Case #{case_idx} — 1D U-Net Inference | Accuracy: {acc:.2f}% | Macro-F1: {macro_f1:.4f}",
        fontsize=11.5, color="#eeeeee", pad=8
    )

    # 2. AWP with Ground-Truth Shading
    axes[1].set_facecolor("#181818")
    axes[1].axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    axes[1].plot(t, awp, color="#f5f5f5", linewidth=0.95, zorder=2)
    _shade_regimes(axes[1], t, y_true, alpha=0.45)
    axes[1].set_ylabel("AWP (hPa)\n[TRUE]", fontsize=9.5, color="#81c784")

    # 3. AWP with U-Net Predicted Shading + Error Highlights
    axes[2].set_facecolor("#181818")
    axes[2].axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    axes[2].plot(t, awp, color="#f5f5f5", linewidth=0.95, zorder=2)
    _shade_regimes(axes[2], t, y_pred, alpha=0.45)
    axes[2].set_ylabel("AWP (hPa)\n[PREDICTED]", fontsize=9.5, color="#ffd54f")

    # Mark misclassified regions along the bottom of panel 3 with bright magenta ticks
    mismatch = y_true != y_pred
    if np.any(mismatch):
        ymin = axes[2].get_ylim()[0]
        axes[2].scatter(t[mismatch], np.full(mismatch.sum(), ymin + 1.0), color="#ff4081", s=4, label="Mismatch", zorder=4)

    # 4. Dynamic Compliance
    axes[3].set_facecolor("#181818")
    axes[3].plot(t, cdyn, color="#4db6ac", linewidth=1.3)
    dot_step = int(8 * fs)
    axes[3].scatter(t[::dot_step], cdyn[::dot_step], color="#80cbc4", s=20)
    axes[3].set_ylabel("Cdyn", fontsize=9)
    axes[3].set_xlabel("Time (seconds)", fontsize=10)

    for ax in axes:
        ax.grid(True, color="#2a2a2a", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color=REGIME_COLORS[k], alpha=0.7, label=regime_names[k]) for k in sorted(regime_names)]
    axes[1].legend(handles=patches, loc="upper right", ncol=5, frameon=True, facecolor="#1e1e1e", fontsize=8.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CHECKPOINT_PATH = "unet_baseline_artifacts/best_unet1d.pt"
    TEST_NPZ_PATH = "anesthesia_synth_dataset/test.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, win_samples, fs, regime_names = load_trained_model(CHECKPOINT_PATH, device)

    test_data = np.load(TEST_NPZ_PATH)
    X_test = test_data["X"]
    y_test = test_data["y"]

    # Run inference and visualize 3 test samples
    for case_idx in [0, 1, 2]:
        signals = X_test[case_idx]   # (3, L)
        y_true = y_test[case_idx]    # (L,)
        y_pred, _ = predict_single_recording(model, signals, mean, std, win_samples, device)

        t = np.arange(len(y_true)) / fs
        co2, awp, cdyn = signals
        plot_inference_comparison(t, co2, awp, cdyn, y_true, y_pred, regime_names, case_idx, fs=fs)