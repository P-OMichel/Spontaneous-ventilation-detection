import os
import json
import math
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================================================================
# 0. CANONICAL REGIME DEFINITIONS & COLOR MAPPING
# ==============================================================================
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

# Mapping from Real AnnotationStore IDs -> Model IDs
# Real:  0=Controlled, 1=Unknown, 2=Apnea, 3=Transition, 4=Efficient Spont
# Model: 0=Controlled, 1=Transition, 2=Efficient Spont, 3=Apnea, 4=Unknown
REAL_TO_MODEL_ID = {
    0: 0,  # Controlled -> Controlled
    1: 4,  # Unknown -> Unknown / Artifact
    2: 3,  # Apnea -> Apnea
    3: 1,  # Transition -> Transition
    4: 2,  # Efficient Spontaneous -> Efficient Spontaneous
}

FEATURE_NAMES = [
    "co2_max", "co2_min", "co2_mean", "co2_std",
    "co2_plateau_width_sec", "co2_cleft_depth",
    "awp_max", "awp_min", "awp_baseline", "awp_neg_dip", "awp_pos_area",
    "cdyn_mean", "cdyn_delta",
    "token_dur_sec", "inst_rr_bpm", "awp_co2_corr",
    "rolling_cv_rr", "rolling_cv_co2"
]
NUM_FEATURES = len(FEATURE_NAMES)


# ==============================================================================
# 1. BREATH & EVENT TOKENIZER (Matches Training Notebook Exactly)
# ==============================================================================
def segment_into_breath_boundaries(co2, awp, fs=128):
    n_samples = len(co2)
    co2_smooth = gaussian_filter1d(co2, sigma=int(0.15 * fs))
    awp_smooth = gaussian_filter1d(awp, sigma=int(0.12 * fs))

    co2_peaks, _ = find_peaks(co2_smooth, height=8.0, distance=int(1.2 * fs), prominence=5.0)
    boundaries = [0]

    if len(co2_peaks) >= 2:
        for i in range(len(co2_peaks) - 1):
            p1, p2 = co2_peaks[i], co2_peaks[i + 1]
            trough = p1 + int(np.argmin(co2_smooth[p1:p2]))
            boundaries.append(trough)
    else:
        awp_peaks, _ = find_peaks(np.abs(awp_smooth), height=3.0, distance=int(1.2 * fs))
        for i in range(len(awp_peaks) - 1):
            mid = (awp_peaks[i] + awp_peaks[i + 1]) // 2
            boundaries.append(mid)

    boundaries.append(n_samples)
    boundaries = sorted(list(set(boundaries)))

    max_token_len = int(3.0 * fs)
    sub_step = int(2.0 * fs)
    final_segments = []

    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s <= 0:
            continue
        if (e - s) > max_token_len:
            curr = s
            while curr < e:
                nxt = min(e, curr + sub_step)
                if e - nxt < int(0.5 * fs):
                    nxt = e
                final_segments.append((curr, nxt))
                curr = nxt
        else:
            final_segments.append((s, e))

    return final_segments


def extract_tokens_for_case(signals_3xL, fs=128):
    co2, awp, cdyn = signals_3xL
    segments = segment_into_breath_boundaries(co2, awp, fs=fs)
    n_tokens = len(segments)

    feats = np.zeros((n_tokens, NUM_FEATURES), dtype=np.float32)
    prev_cdyn = cdyn[0]

    for i, (s, e) in enumerate(segments):
        c_seg = co2[s:e]
        a_seg = awp[s:e]
        d_seg = cdyn[s:e]
        dur_s = max(1e-3, (e - s) / fs)

        c_max = float(np.max(c_seg))
        c_min = float(np.min(c_seg))
        c_mean = float(np.mean(c_seg))
        c_std = float(np.std(c_seg))

        if c_max > 8.0:
            high_mask = c_seg > (0.75 * c_max)
            plateau_w = float(np.sum(high_mask)) / fs
            active_idx = np.where(c_seg > 0.4 * c_max)[0]
            if len(active_idx) > int(0.5 * fs):
                mid_slice = c_seg[active_idx[0] + int(0.15 * fs) : active_idx[-1] - int(0.15 * fs)]
                cleft_depth = float(c_max - np.min(mid_slice)) if len(mid_slice) > 0 else 0.0
            else:
                cleft_depth = 0.0
        else:
            plateau_w = 0.0
            cleft_depth = 0.0

        a_max = float(np.max(a_seg))
        a_min = float(np.min(a_seg))
        a_base = float(np.percentile(a_seg, 20))
        a_neg_dip = float(a_min - a_base)
        a_pos_area = float(np.sum(np.maximum(0.0, a_seg - a_base))) / fs

        d_mean = float(np.mean(d_seg))
        d_delta = float(d_mean - prev_cdyn)
        prev_cdyn = d_mean
        inst_rr = 60.0 / dur_s if c_max > 8.0 or (a_max - a_min) > 3.0 else 0.0

        if np.std(a_seg) > 0.3 and np.std(c_seg) > 0.5:
            corr = float(np.corrcoef(a_seg, c_seg)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        feats[i, :16] = [
            c_max, c_min, c_mean, c_std, plateau_w, cleft_depth,
            a_max, a_min, a_base, a_neg_dip, a_pos_area,
            d_mean, d_delta, dur_s, inst_rr, corr
        ]

    for i in range(n_tokens):
        w_start = max(0, i - 2)
        w_end = min(n_tokens, i + 3)
        rr_win = feats[w_start:w_end, 14]
        co2_win = feats[w_start:w_end, 0]
        feats[i, 16] = float(np.std(rr_win) / (np.mean(rr_win) + 1.0))
        feats[i, 17] = float(np.std(co2_win) / (np.mean(co2_win) + 1.0))

    return feats, segments


# ==============================================================================
# 2. MODEL ARCHITECTURES (UNet1D, BreathBiGRU, BreathTransformer)
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


class BreathBiGRU(nn.Module):
    def __init__(self, in_features=NUM_FEATURES, hidden_dim=96, num_layers=2, num_classes=5, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, pad_mask=None):
        h = self.input_proj(x)
        out, _ = self.gru(h)
        return self.head(out)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BreathTransformer(nn.Module):
    def __init__(self, in_features=NUM_FEATURES, d_model=128, nhead=4, num_layers=3, num_classes=5, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=1500)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x, pad_mask=None):
        h = self.pos_enc(self.input_proj(x))
        out = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(out)


# ==============================================================================
# 3. UNIFIED CHECKPOINT LOADER & MULTI-MODEL INFERENCE ENGINE
# ==============================================================================
def load_any_model(checkpoint_path, device):
    """
    Automatically detects model type ('UNet1D', 'BreathBiGRU', or 'BreathTransformer')
    from the checkpoint metadata and returns a unified inference wrapper dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name", "UNet1D")
    cfg = ckpt["model_config"]

    if model_name == "UNet1D":
        model = UNet1D(
            in_channels=cfg["in_channels"],
            num_classes=cfg["num_classes"],
            base_filters=cfg["base_filters"],
            dropout=0.0,
        ).to(device)
        norm_mean = np.array(ckpt["normalization"]["mean"], dtype=np.float32)
        norm_std = np.array(ckpt["normalization"]["std"], dtype=np.float32)
        extra = {"window_samples": cfg["window_samples"]}

    elif model_name == "BreathBiGRU":
        model = BreathBiGRU(
            in_features=cfg["in_features"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_classes=cfg["num_classes"],
            dropout=0.0,
        ).to(device)
        norm_mean = np.array(ckpt["feat_mean"], dtype=np.float32)
        norm_std = np.array(ckpt["feat_std"], dtype=np.float32)
        extra = {}

    elif model_name == "BreathTransformer":
        model = BreathTransformer(
            in_features=cfg["in_features"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            num_classes=cfg["num_classes"],
            dropout=0.0,
        ).to(device)
        norm_mean = np.array(ckpt["feat_mean"], dtype=np.float32)
        norm_std = np.array(ckpt["feat_std"], dtype=np.float32)
        extra = {}

    else:
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "name": model_name,
        "model": model,
        "mean": norm_mean,
        "std": norm_std,
        "fs": ckpt.get("fs", 128),
        "extra": extra,
    }


@torch.no_grad()
def predict_recording_with_bundle(bundle, signals_3xL, device, stride_ratio=0.25):
    """
    Runs inference for any loaded model bundle (UNet1D, BreathBiGRU, or BreathTransformer)
    and returns sample-level predictions (L,) and probabilities (5, L) at 128 Hz.
    """
    model_name = bundle["name"]
    model = bundle["model"]
    mean = bundle["mean"]
    std = bundle["std"]
    fs = bundle["fs"]
    _, total_len = signals_3xL.shape

    if model_name == "UNet1D":
        window_samples = bundle["extra"]["window_samples"]
        num_classes = model.head.out_channels
        norm_sig = (signals_3xL - mean[:, None]) / (std[:, None] + 1e-6)

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
        return np.argmax(avg_probs, axis=0), avg_probs

    else:
        # Token-based inference (BreathBiGRU or BreathTransformer)
        feats, segments = extract_tokens_for_case(signals_3xL, fs=fs)
        norm_f = (feats - mean) / (std + 1e-6)
        xb = torch.from_numpy(norm_f).unsqueeze(0).float().to(device)
        logits = model(xb)
        tok_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (N_tok, 5)
        tok_preds = np.argmax(tok_probs, axis=-1)

        pred_labels = np.zeros(total_len, dtype=np.int64)
        pred_probs = np.zeros((tok_probs.shape[1], total_len), dtype=np.float32)
        for tok_idx, (s, e) in enumerate(segments):
            pred_labels[s:e] = tok_preds[tok_idx]
            pred_probs[:, s:e] = tok_probs[tok_idx][:, None]

        return pred_labels, pred_probs


# ==============================================================================
# 4. REAL DATA LOADER (*_vent_annotated.npz)
# ==============================================================================
def _estimate_surrogate_cdyn(awp, fs, default_cdyn=55.0):
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
    if win_labels is None or len(t_windows) == 0:
        return None
    y_samples = np.full(len(t_samples), fill_value=4, dtype=np.int64)
    half_step = 0.5 * step_sec
    for tw, raw_id in zip(t_windows, win_labels):
        mapped_id = REAL_TO_MODEL_ID.get(int(raw_id), 4)
        mask = (t_samples >= (tw - half_step)) & (t_samples < (tw + half_step))
        y_samples[mask] = mapped_id
    return y_samples


def load_real_annotated_case(annotated_npz_path, use_cropped=True, default_cdyn=55.0):
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

    co2 = np.nan_to_num(co2, nan=0.0).astype(np.float32)
    awp = np.nan_to_num(awp, nan=0.0).astype(np.float32)
    cdyn = _estimate_surrogate_cdyn(awp, fs=fs, default_cdyn=default_cdyn)

    signals_3xL = np.stack([co2, awp, cdyn], axis=0)
    y_true_128hz = _upsample_window_labels_to_samples(t, t_win, regimes, step_sec=step_sec)

    return t, signals_3xL, y_true_128hz, fs, case_label


# ==============================================================================
# 5. MULTI-MODEL VISUALIZATION (CO2 + Ground Truth + All Loaded Models + Cdyn)
# ==============================================================================
def _shade_regimes(ax, t, labels, alpha=0.45):
    change_idx = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(labels) - 1]))
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[e], color=REGIME_COLORS[int(labels[s])], alpha=alpha, zorder=1)
        if s > 0:
            ax.axvline(t[s], color="#aaaaaa", linestyle=":", linewidth=0.8, alpha=0.7)


def plot_multi_model_comparison(t, co2, awp, cdyn, y_true, preds_dict, case_title, fs=128):
    """
    Plots CO2, optional Ground Truth AWP shading, stacked AWP prediction tracks
    for every loaded model (UNet1D, BreathBiGRU, BreathTransformer), and Cdyn.
    """
    has_truth = y_true is not None
    model_names = list(preds_dict.keys())
    n_models = len(model_names)

    n_rows = 2 + (1 if has_truth else 0) + n_models
    height_ratios = [1.05] + ([1.35] if has_truth else []) + [1.35] * n_models + [0.6]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(16, 3.0 + 2.2 * n_rows), sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.22}
    )
    fig.patch.set_facecolor("#121212")

    # 1. CO2 Panel
    axes[0].set_facecolor("#181818")
    axes[0].plot(t, co2, color="#29b6f6", linewidth=1.15)
    axes[0].set_ylabel("CO2 (mmHg)", fontsize=9.5)
    axes[0].set_title(f"{case_title} — Multi-Model Inference Comparison", fontsize=12, color="#eeeeee", pad=8)

    row_cursor = 1

    # 2. Ground Truth Panel (if available)
    if has_truth:
        ax_gt = axes[row_cursor]
        ax_gt.set_facecolor("#181818")
        ax_gt.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
        ax_gt.plot(t, awp, color="#f5f5f5", linewidth=0.95, zorder=2)
        _shade_regimes(ax_gt, t, y_true, alpha=0.45)
        ax_gt.set_ylabel("AWP (hPa)\n[GROUND TRUTH]", fontsize=9, color="#81c784")
        row_cursor += 1

    # 3. One Prediction Panel per Model
    for m_name in model_names:
        y_pred = preds_dict[m_name]
        ax_m = axes[row_cursor]
        ax_m.set_facecolor("#181818")
        ax_m.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
        ax_m.plot(t, awp, color="#f5f5f5", linewidth=0.95, zorder=2)
        _shade_regimes(ax_m, t, y_pred, alpha=0.45)

        if has_truth:
            acc = (y_true == y_pred).mean() * 100.0
            present = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
            mf1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
            ax_m.set_ylabel(f"{m_name}\nAcc: {acc:.1f}%\nF1: {mf1:.3f}", fontsize=9, color="#ffd54f")

            mismatch = y_true != y_pred
            if np.any(mismatch):
                ymin = ax_m.get_ylim()[0]
                ax_m.scatter(t[mismatch], np.full(mismatch.sum(), ymin + 0.8), color="#ff4081", s=3.5, zorder=4)
        else:
            ax_m.set_ylabel(f"{m_name}\n[PREDICTED]", fontsize=9, color="#ffd54f")

        row_cursor += 1

    # 4. Dynamic Compliance Panel
    ax_cdyn = axes[row_cursor]
    ax_cdyn.set_facecolor("#181818")
    ax_cdyn.plot(t, cdyn, color="#4db6ac", linewidth=1.3)
    dot_step = max(1, int(8 * fs))
    ax_cdyn.scatter(t[::dot_step], cdyn[::dot_step], color="#80cbc4", s=20)
    ax_cdyn.set_ylabel("Cdyn", fontsize=9)
    ax_cdyn.set_xlabel("Time (seconds)", fontsize=10)

    for ax in axes:
        ax.grid(True, color="#2a2a2a", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    patches = [mpatches.Patch(color=REGIME_COLORS[k], alpha=0.7, label=CANONICAL_NAMES[k]) for k in sorted(CANONICAL_NAMES)]
    axes[1].legend(handles=patches, loc="upper right", ncol=5, frameon=True, facecolor="#1e1e1e", fontsize=8.5)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 6. MAIN ENTRYPOINT: EVALUATE ALL SAVED MODELS ON SYNTHETIC OR REAL DATA
# ==============================================================================
if __name__ == "__main__":
    # Directory containing your downloaded checkpoints (no 'artifacts' suffix)
    MODELS_DIR = "ventilation_models"
    CHECKPOINT_FILES = [
        os.path.join(MODELS_DIR, "best_unet1d.pt"),
        os.path.join(MODELS_DIR, "best_breathbigru.pt"),
        os.path.join(MODELS_DIR, "best_breathtransformer.pt"),
    ]

    # Choose data source: "synthetic" or "real"
    DATA_SOURCE = "real"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all available model checkpoints from MODELS_DIR
    loaded_bundles = []
    for ckpt_path in CHECKPOINT_FILES:
        if os.path.exists(ckpt_path):
            bundle = load_any_model(ckpt_path, device)
            loaded_bundles.append(bundle)
            print(f"Loaded model: {bundle['name']} <- {ckpt_path}")
        else:
            print(f"Skipping missing checkpoint: {ckpt_path}")

    if not loaded_bundles:
        raise FileNotFoundError(f"No model checkpoints found in '{MODELS_DIR}'.")

    if DATA_SOURCE == "synthetic":
        TEST_NPZ_PATH = "anesthesia_synth_dataset/test.npz"
        test_data = np.load(TEST_NPZ_PATH)
        X_test, y_test = test_data["X"], test_data["y"]
        fs = loaded_bundles[0]["fs"]

        for case_idx in [0, 1, 2]:
            signals = X_test[case_idx]
            y_true = y_test[case_idx]
            t = np.arange(len(y_true)) / fs

            preds_dict = {}
            print(f"\n==================== Synthetic Test Case #{case_idx} ====================")
            for bundle in loaded_bundles:
                y_pred, _ = predict_recording_with_bundle(bundle, signals, device)
                preds_dict[bundle["name"]] = y_pred
                acc = (y_true == y_pred).mean() * 100.0
                mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                print(f"  {bundle['name']:<20} -> Accuracy: {acc:.2f}% | Macro-F1: {mf1:.4f}")

            co2, awp, cdyn = signals
            plot_multi_model_comparison(
                t, co2, awp, cdyn, y_true, preds_dict,
                case_title=f"Synthetic Test Case #{case_idx}", fs=fs
            )

    elif DATA_SOURCE == "real":
        REAL_FILES = [
            r"db_cases_128hz/vitaldb_case_1_vent_annotated.npz",
            # Add additional real cases here:
            # r"db_cases_128hz/vitaldb_case_2_vent_annotated.npz",
        ]
        USE_CROPPED_SEGMENT = True

        for file_path in REAL_FILES:
            t, signals, y_true, fs, case_label = load_real_annotated_case(
                file_path, use_cropped=USE_CROPPED_SEGMENT, default_cdyn=55.0
            )

            preds_dict = {}
            print(f"\n==================== Real VitalDB: {case_label} ====================")
            for bundle in loaded_bundles:
                y_pred, _ = predict_recording_with_bundle(bundle, signals, device)
                preds_dict[bundle["name"]] = y_pred

                if y_true is not None:
                    present = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
                    acc = (y_true == y_pred).mean() * 100.0
                    mf1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
                    print(f"\n--- {bundle['name']} (Acc: {acc:.2f}% | Macro-F1: {mf1:.4f}) ---")
                    print(classification_report(
                        y_true, y_pred,
                        labels=present,
                        target_names=[CANONICAL_NAMES[i] for i in present],
                        digits=4,
                        zero_division=0
                    ))

            co2, awp, cdyn = signals
            plot_multi_model_comparison(
                t, co2, awp, cdyn, y_true, preds_dict,
                case_title=f"Real VitalDB: {case_label}", fs=fs
            )