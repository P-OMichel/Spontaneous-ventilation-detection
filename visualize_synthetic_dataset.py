import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REGIME_COLORS = {
    0: "#1f4e78",  # Controlled (Steel Blue)
    1: "#7f6000",  # Transition (Amber / Brown)
    2: "#1e5631",  # Efficient Spontaneous (Dark Green)
    3: "#661d1d",  # Apnea (Deep Red)
    4: "#4a4a4a",  # Unknown / Artifact (Charcoal Grey)
}


def load_split(dataset_dir="anesthesia_synth_dataset", split="train"):
    """Loads X, y, and metadata from a saved dataset directory."""
    meta_path = os.path.join(dataset_dir, "metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    npz_path = os.path.join(dataset_dir, f"{split}.npz")
    data = np.load(npz_path)
    X = data["X"]  # Shape: (N_cases, 3, N_samples) -> [CO2, AWP, Cdyn]
    y = data["y"]  # Shape: (N_cases, N_samples)
    regime_names = {int(k): v for k, v in metadata["regimes"].items()}
    return X, y, metadata["fs"], regime_names


def plot_saved_case(dataset_dir="anesthesia_synth_dataset", split="train", case_idx=0, t_range=None):
    """
    Plots a single saved case (CO2, AWP with regime shading, and Cdyn) from disk.
    """
    X, y, fs, regime_names = load_split(dataset_dir, split)
    co2, awp, cdyn = X[case_idx]
    labels = y[case_idx]
    t = np.arange(len(labels)) / fs

    if t_range is not None:
        mask = (t >= t_range[0]) & (t <= t_range[1])
        t, co2, awp, cdyn, labels = t[mask], co2[mask], awp[mask], cdyn[mask], labels[mask]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.75, 0.7], "hspace": 0.18}
    )
    fig.patch.set_facecolor("#121212")

    # 1. Top Panel: Capnography CO2
    ax_co2 = axes[0]
    ax_co2.set_facecolor("#181818")
    ax_co2.plot(t, co2, color="#29b6f6", linewidth=1.25)
    ax_co2.set_ylabel("CO2 (mmHg)", fontsize=10, color="#cccccc")
    ax_co2.set_title(f"[{split.upper()} | Case #{case_idx}] Capnography CO2 ({fs} Hz)", fontsize=11, color="#eeeeee", pad=6)
    ax_co2.set_ylim(-2, max(58, np.max(co2) * 1.15))
    ax_co2.grid(True, color="#2e2e2e", linestyle="-", linewidth=0.6)

    # 2. Middle Panel: Airway Pressure AWP + Regime Shading
    ax_awp = axes[1]
    ax_awp.set_facecolor("#181818")
    ax_awp.axhline(0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
    ax_awp.plot(t, awp, color="#f5f5f5", linewidth=1.0, zorder=3)
    ax_awp.set_ylabel("AWP (hPa)", fontsize=10, color="#cccccc")
    ax_awp.set_title("Airway Pressure AWP — Shaded by Saved Ground-Truth Regime", fontsize=11, color="#eeeeee", pad=6)
    ax_awp.grid(True, color="#2e2e2e", linestyle="-", linewidth=0.6, zorder=0)

    change_idx = np.where(np.diff(labels) != 0)[0] + 1
    seg_starts = np.concatenate(([0], change_idx))
    seg_ends = np.concatenate((change_idx, [len(labels) - 1]))

    for s, e in zip(seg_starts, seg_ends):
        state = int(labels[s])
        color = REGIME_COLORS[state]
        ax_awp.axvspan(t[s], t[e], color=color, alpha=0.45, zorder=2)
        ax_co2.axvspan(t[s], t[e], color=color, alpha=0.15, zorder=0)
        if s > 0:
            ax_awp.axvline(t[s], color="#aaaaaa", linestyle=":", linewidth=0.8, alpha=0.7)

    patches = [
        mpatches.Patch(color=REGIME_COLORS[k], alpha=0.65, label=regime_names[k])
        for k in sorted(regime_names)
    ]
    ax_awp.legend(
        handles=patches, loc="upper right", ncol=5, frameon=True,
        facecolor="#1e1e1e", edgecolor="#444444", fontsize=8.5
    )

    # 3. Bottom Panel: Sample-and-Hold Dynamic Lung Compliance
    ax_cdyn = axes[2]
    ax_cdyn.set_facecolor("#181818")
    ax_cdyn.plot(t, cdyn, color="#4db6ac", linewidth=1.4)
    dot_step = int(8 * fs)
    ax_cdyn.scatter(t[::dot_step], cdyn[::dot_step], color="#80cbc4", s=22, zorder=4)
    ax_cdyn.set_ylabel("Cdyn\n(mL/mbar)", fontsize=9, color="#cccccc")
    ax_cdyn.set_xlabel("Time (seconds)", fontsize=10, color="#cccccc")
    ax_cdyn.set_title("Dynamic Lung Compliance", fontsize=10, color="#eeeeee", pad=5)
    ax_cdyn.set_ylim(0, max(95, np.max(cdyn) * 1.2))
    ax_cdyn.grid(True, color="#2e2e2e", linestyle="-", linewidth=0.6)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")
        ax.tick_params(colors="#aaaaaa", labelsize=9)

    plt.tight_layout()
    plt.show()


def plot_saved_gallery(dataset_dir="anesthesia_synth_dataset", split="test", case_indices=(0, 1, 2, 3)):
    """
    Loads 4 saved cases from disk and displays a 2x2 overview grid plus
    a bottom row zooming into each case's Transition CO2 top-plateau shapes.
    """
    X, y, fs, regime_names = load_split(dataset_dir, split)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#121212")

    outer_gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.35, wspace=0.14)
    zoom_gs = outer_gs[2, :].subgridspec(1, 4, wspace=0.14)

    for idx, case_idx in enumerate(case_indices[:4]):
        co2, awp, _ = X[case_idx]
        labels = y[case_idx]
        t = np.arange(len(labels)) / fs

        row, col = divmod(idx, 2)
        inner_gs = outer_gs[row, col].subgridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.08)

        ax_co2 = fig.add_subplot(inner_gs[0])
        ax_awp = fig.add_subplot(inner_gs[1], sharex=ax_co2)

        for ax in (ax_co2, ax_awp):
            ax.set_facecolor("#181818")
            ax.grid(True, color="#2a2a2a", linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#444444")
            ax.spines["bottom"].set_color("#444444")
            ax.tick_params(colors="#aaaaaa", labelsize=8)

        ax_co2.plot(t, co2, color="#29b6f6", linewidth=1.0)
        ax_co2.set_ylabel("CO2", fontsize=8.5, color="#cccccc")
        ax_co2.set_ylim(-2, max(56, np.max(co2) * 1.12))
        ax_co2.set_title(f"{split.upper()} Case #{case_idx} — Full Trajectory", fontsize=10, color="#eeeeee", pad=4)
        plt.setp(ax_co2.get_xticklabels(), visible=False)

        ax_awp.axhline(0, color="#888888", linestyle="--", linewidth=0.7)
        ax_awp.plot(t, awp, color="#f5f5f5", linewidth=0.85)
        ax_awp.set_ylabel("AWP", fontsize=8.5, color="#cccccc")
        ax_awp.set_xlabel("Time (s)", fontsize=8.5, color="#aaaaaa")

        change_idx = np.where(np.diff(labels) != 0)[0] + 1
        seg_starts = np.concatenate(([0], change_idx))
        seg_ends = np.concatenate((change_idx, [len(labels) - 1]))

        for s, e in zip(seg_starts, seg_ends):
            color = REGIME_COLORS[int(labels[s])]
            ax_awp.axvspan(t[s], t[e], color=color, alpha=0.45)
            ax_co2.axvspan(t[s], t[e], color=color, alpha=0.15)

        # Bottom row: 35-second zoom into Transition CO2 plateaus
        trans_indices = np.where(labels == 1)[0]
        ax_zoom = fig.add_subplot(zoom_gs[0, idx])
        ax_zoom.set_facecolor("#181818")
        ax_zoom.grid(True, color="#2a2a2a", linewidth=0.5)

        if len(trans_indices) > 0:
            start_i = trans_indices[0]
            end_i = min(len(t) - 1, start_i + int(35 * fs))
            ax_zoom.plot(t[start_i:end_i], co2[start_i:end_i], color="#29b6f6", linewidth=1.3)
            ax_zoom.axvspan(t[start_i], t[end_i], color=REGIME_COLORS[1], alpha=0.2)
            ax_zoom.set_ylim(-2, 58)
            ax_zoom.set_title(f"Case #{case_idx}: Transition CO2 Zoom (35s)", fontsize=9, color="#ffd54f")
            ax_zoom.set_xlabel("Time (s)", fontsize=8, color="#aaaaaa")
            if idx == 0:
                ax_zoom.set_ylabel("CO2 (mmHg)", fontsize=8.5, color="#cccccc")

        ax_zoom.spines["top"].set_visible(False)
        ax_zoom.spines["right"].set_visible(False)
        ax_zoom.spines["left"].set_color("#444444")
        ax_zoom.spines["bottom"].set_color("#444444")
        ax_zoom.tick_params(colors="#aaaaaa", labelsize=8)

    patches = [
        mpatches.Patch(color=REGIME_COLORS[k], alpha=0.7, label=regime_names[k])
        for k in sorted(regime_names)
    ]
    fig.legend(
        handles=patches, loc="upper center", ncol=5, frameon=True,
        facecolor="#1e1e1e", edgecolor="#444444", fontsize=9.5, bbox_to_anchor=(0.5, 0.985)
    )
    plt.show()


if __name__ == "__main__":
    # 1. Inspect a single saved training case in full detail (CO2 + AWP + Cdyn)
    plot_saved_case(dataset_dir="anesthesia_synth_dataset", split="train", case_idx=0)

    # 2. Inspect 4 saved test cases side-by-side with zoomed Transition plateaus
    plot_saved_gallery(dataset_dir="anesthesia_synth_dataset", split="test", case_indices=(0, 1, 2, 3))