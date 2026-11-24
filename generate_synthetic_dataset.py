import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REGIME_NAMES = {
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


def generate_synthetic_case(duration_sec=550, fs=128, seed=None):
    """
    Generates coupled synthetic anesthesia waveforms (CO2, AWP, Cdyn) modeled on
    real VitalDB / Primus emergence trajectories, including:
      - Wide, flat or upward-sloping CO2 top plateaus (breath-holds / slow expiration)
      - Deep mid-plateau curare clefts & double-humped plateaus
      - Incomplete-washout stacked breaths ('W'/'M' clusters)
      - Post-pause hypercapnic CO2 overshoot
      - Manual bag squeeze spikes & sudden Cdyn step drops
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs

    # Randomize baseline patient physiology
    R_rs = rng.uniform(5.0, 14.0)         # Airway resistance (hPa / (L/s))
    C_rs = rng.uniform(0.048, 0.080)      # Passive lung compliance (L / hPa)
    peep_set = rng.choice([0.0, 3.0, 5.0])
    co2_lag_samples = int(rng.uniform(1.0, 2.2) * fs)
    alv_co2_base = rng.uniform(36.0, 42.0)

    # 1. Build a rich, realistic clinical trajectory (mirrors your 5 screenshots)
    labels = np.zeros(n_samples, dtype=int)
    cursor = 0
    trajectory_templates = [
        # Template A: Smooth emergence with intermittent apnea & final extubation (like 3600-3800 & 20340-20900)
        [
            (0, rng.uniform(110, 160)),  # Controlled
            (1, rng.uniform(45, 75)),    # Transition (bagging + wide clefted plateaus)
            (2, rng.uniform(80, 130)),   # Efficient Spontaneous
            (3, rng.uniform(15, 28)),    # Brief Apnea
            (2, rng.uniform(12, 22)),    # Pre-extubation cough/breath cluster
            (3, rng.uniform(25, 45)),    # Post-extubation Apnea
        ],
        # Template B: Multi-stage emergence with rescue bagging & artifact (like 14600-15150 & 10600-10900)
        [
            (0, rng.uniform(90, 130)),   # Controlled
            (1, rng.uniform(45, 70)),    # Transition 1
            (3, rng.uniform(18, 30)),    # Apnea pause
            (4, rng.uniform(12, 22)),    # Unknown / Mask leak or suctioning
            (1, rng.uniform(35, 60)),    # Transition 2 (rescue bagging + long plateaus)
            (2, rng.uniform(70, 110)),   # Efficient Spontaneous
            (3, rng.uniform(25, 50)),    # Extubation Apnea
        ],
    ]
    sequence = trajectory_templates[rng.integers(0, len(trajectory_templates))]

    for state, dur in sequence:
        end = min(n_samples, cursor + int(dur * fs))
        labels[cursor:end] = state
        cursor = end
        if cursor >= n_samples:
            break

    # Direct waveform buffers (allows explicit control of top-plateau shapes)
    awp = np.zeros(n_samples)
    co2 = np.zeros(n_samples)
    vt_proxy = np.zeros(n_samples)   # Tracks delivered/inhaled tidal volume for Cdyn
    pmus_proxy = np.zeros(n_samples) # Tracks patient muscular effort

    # 2. Synthesize breath-by-breath waveforms per regime
    idx = 0
    hypercapnia_offset = 0.0  # Builds up during apnea/hypoventilation, washes out during good breathing

    while idx < n_samples:
        state = labels[idx]

        # --- STATE 0: CONTROLLED VENTILATION (VCV) ---
        if state == 0:
            rr = rng.uniform(11.5, 14.5)
            cycle_len = int((60.0 / rr) * fs)
            end = min(n_samples, idx + cycle_len)
            dur_s = (end - idx) / fs
            t_loc = np.linspace(0, dur_s, end - idx)

            ti = 0.32 * dur_s
            vt = rng.uniform(0.45, 0.55)
            flow_insp = vt / ti

            # AWP: Classic VCV sawtooth rise to Ppeak + PEEP
            awp_cycle = np.where(
                t_loc <= ti,
                peep_set + R_rs * flow_insp + (flow_insp * t_loc / C_rs),
                peep_set + (vt / C_rs) * np.exp(-(t_loc - ti) / (R_rs * C_rs)) * 0.4,
            )
            awp[idx:end] = awp_cycle

            # CO2: Uniform flat-topped square-wave plateau
            # Note: CO2 rises during expiration (after ti) and drops to 0 during next inspiration
            co2_target = alv_co2_base + hypercapnia_offset
            hypercapnia_offset = max(0.0, hypercapnia_offset - 1.5)

            t_exp = np.maximum(0.0, t_loc - ti)
            exp_dur = max(0.2, dur_s - ti)
            # Crisp S-upstroke, completely flat top plateau, sharp drop at end of expiration
            upstroke = 1.0 - np.exp(-t_exp / 0.14)
            washout = 1.0 / (1.0 + np.exp((t_exp - (exp_dur - 0.18)) / 0.03))
            co2[idx:end] = np.where(t_loc <= ti, 0.0, co2_target * upstroke * washout)

            vt_proxy[idx:end] = vt * 1000.0
            idx = end

        # --- STATE 1: TRANSITION (5 Real Clinical Top-Plateau Sub-Patterns) ---
        elif state == 1:
            sub_pattern = rng.choice(
                ["wide_plateau", "curare_cleft", "stacked_cluster", "manual_bagging"],
                p=[0.30, 0.30, 0.25, 0.15],
            )

            if sub_pattern == "wide_plateau":
                # Pattern 1 (like 3725-3737 & 14875-14893): Prolonged 4.5-8.5s flat or rising top plateau
                dur_s = rng.uniform(4.5, 8.5)
                end = min(n_samples, idx + int(dur_s * fs))
                t_loc = np.linspace(0, (end - idx) / fs, end - idx)
                ti = rng.uniform(0.45, 0.75)

                # Small negative inspiratory gasp on AWP, then quiet baseline during breath-hold
                awp[idx:end] = np.where(
                    t_loc <= ti,
                    -rng.uniform(2.5, 5.5) * np.sin(np.pi * t_loc / ti),
                    rng.uniform(0.0, 1.5),
                )

                # CO2 stays pinned at the top plateau for several seconds, climbing slightly
                t_exp = np.maximum(0.0, t_loc - ti)
                exp_dur = max(0.5, (end - idx) / fs - ti)
                upstroke = 1.0 - np.exp(-t_exp / 0.18)
                # Gradual upward slope across the long top plateau (+4 to +9 mmHg)
                plateau_ramp = (alv_co2_base + hypercapnia_offset) + rng.uniform(4.0, 9.0) * (t_exp / exp_dur)
                washout = 1.0 / (1.0 + np.exp((t_exp - (exp_dur - 0.15)) / 0.03))
                co2[idx:end] = np.where(t_loc <= ti, 0.0, plateau_ramp * upstroke * washout)

                hypercapnia_offset = min(10.0, hypercapnia_offset + 2.5)
                pmus_proxy[idx:end] = 4.0
                idx = end

            elif sub_pattern == "curare_cleft":
                # Pattern 2 (like 20615-20645 & 5195-5205): Broad top plateau split by 1 or 2 sharp V-clefts
                dur_s = rng.uniform(3.8, 6.5)
                end = min(n_samples, idx + int(dur_s * fs))
                t_loc = np.linspace(0, (end - idx) / fs, end - idx)
                ti = rng.uniform(0.45, 0.70)

                t_exp = np.maximum(0.0, t_loc - ti)
                exp_dur = max(0.5, (end - idx) / fs - ti)
                upstroke = 1.0 - np.exp(-t_exp / 0.15)
                washout = 1.0 / (1.0 + np.exp((t_exp - (exp_dur - 0.15)) / 0.03))
                base_top = alv_co2_base + hypercapnia_offset + rng.uniform(1.0, 5.0)

                # Carve 1 or 2 deep downward clefts into the top plateau
                cleft_factor = np.ones_like(t_loc)
                awp_cleft = np.zeros_like(t_loc)
                n_clefts = rng.choice([1, 2], p=[0.7, 0.3])
                for c in range(n_clefts):
                    c_center = ti + exp_dur * rng.uniform(0.28 + 0.3 * c, 0.50 + 0.25 * c)
                    c_width = rng.uniform(0.14, 0.26)
                    c_depth = rng.uniform(0.45, 0.85)  # Plunges 45-85% down from plateau top
                    notch = c_depth * np.exp(-((t_loc - c_center) ** 2) / (2 * c_width ** 2))
                    cleft_factor -= notch
                    # Corresponding negative diaphragmatic dip on AWP
                    awp_cleft -= rng.uniform(2.0, 4.5) * np.exp(-((t_loc - c_center) ** 2) / (2 * c_width ** 2))

                awp[idx:end] = (
                    np.where(t_loc <= ti, -rng.uniform(2.0, 4.5) * np.sin(np.pi * t_loc / ti), 0.5)
                    + awp_cleft
                )
                co2[idx:end] = np.where(
                    t_loc <= ti, 0.0, base_top * upstroke * washout * np.maximum(0.1, cleft_factor)
                )
                pmus_proxy[idx:end] = 5.0
                idx = end

            elif sub_pattern == "stacked_cluster":
                # Pattern 3 (like 3685-3715 & 14685-14714): Rapid, erratic breaths with incomplete washout
                n_breaths = rng.integers(3, 6)
                for b in range(n_breaths):
                    if idx >= n_samples or labels[idx] != 1:
                        break
                    b_dur = rng.uniform(1.4, 2.5)
                    end = min(n_samples, idx + int(b_dur * fs))
                    t_loc = np.linspace(0, (end - idx) / fs, end - idx)
                    ti = 0.35 * b_dur

                    # Incomplete inspiratory washout: baseline stays elevated at 6-18 mmHg
                    elevated_base = rng.uniform(6.0, 18.0) if b > 0 else 0.0
                    peak_h = rng.uniform(24.0, 42.0)

                    awp[idx:end] = np.where(
                        t_loc <= ti,
                        -rng.uniform(1.5, 3.5) * np.sin(np.pi * t_loc / ti),
                        rng.uniform(0.5, 2.5) * np.sin(np.pi * np.maximum(0, t_loc - ti) / (b_dur - ti)),
                    )
                    wave = np.sin(np.pi * np.clip((t_loc - ti * 0.5) / (b_dur - ti * 0.5), 0, 1))
                    co2[idx:end] = elevated_base + (peak_h - elevated_base) * (wave ** 0.7)
                    pmus_proxy[idx:end] = 3.5
                    idx = end

            else:  # "manual_bagging"
                # Pattern 4 (like 3664-3680 & 5131-5146): High-pressure bag squeezes + asynchronous CO2
                n_squeezes = rng.integers(2, 5)
                for _ in range(n_squeezes):
                    if idx >= n_samples or labels[idx] != 1:
                        break
                    b_dur = rng.uniform(2.0, 3.4)
                    end = min(n_samples, idx + int(b_dur * fs))
                    t_loc = np.linspace(0, (end - idx) / fs, end - idx)
                    ti = rng.uniform(0.5, 0.8)

                    # Large manual bag spike (22-42 hPa) + pre-trigger dip
                    bag_peak = rng.uniform(22.0, 42.0)
                    p_bag = bag_peak * np.exp(-((t_loc - ti) ** 2) / 0.035)
                    p_dip = rng.uniform(2.5, 5.5) * np.exp(-((t_loc - ti * 0.4) ** 2) / 0.01)
                    awp[idx:end] = rng.uniform(0.5, 3.0) + p_bag - p_dip

                    # Irregular CO2 peak
                    t_exp = np.maximum(0.0, t_loc - ti)
                    exp_dur = max(0.3, b_dur - ti)
                    co2_amp = rng.uniform(28.0, 46.0)
                    upstroke = 1.0 - np.exp(-t_exp / 0.16)
                    washout = 1.0 / (1.0 + np.exp((t_exp - (exp_dur - 0.15)) / 0.04))
                    co2[idx:end] = np.where(t_loc <= ti, 0.0, co2_amp * upstroke * washout)

                    vt_proxy[idx:end] = rng.uniform(350, 500)
                    idx = end

        # --- STATE 2: EFFICIENT SPONTANEOUS VENTILATION ---
        elif state == 2:
            # Pattern (like 20668-20822 & 10785-10842): Steady rhythm, rounded dome/plateau, 0 hPa AWP ripples
            rr = rng.uniform(15.0, 20.0)
            cycle_len = int((60.0 / rr) * rng.normal(1.0, 0.06) * fs)
            end = min(n_samples, idx + max(int(1.6 * fs), cycle_len))
            dur_s = (end - idx) / fs
            t_loc = np.linspace(0, dur_s, end - idx)
            ti = 0.38 * dur_s

            # Symmetrical low-amplitude sinusoidal pressure swings around 0 hPa (+/- 2 to 3.5 hPa)
            amp_awp = rng.uniform(2.0, 3.5)
            awp[idx:end] = np.where(
                t_loc <= ti,
                -amp_awp * np.sin(np.pi * t_loc / ti),
                (amp_awp * 0.85) * np.sin(np.pi * (t_loc - ti) / (dur_s - ti)),
            )

            # Rounded / slightly domed full-height alveolar plateau returning cleanly to 0
            co2_target = alv_co2_base + hypercapnia_offset
            hypercapnia_offset = max(0.0, hypercapnia_offset - 1.2)

            t_exp = np.maximum(0.0, t_loc - ti)
            exp_dur = max(0.2, dur_s - ti)
            upstroke = 1.0 - np.exp(-t_exp / 0.16)
            dome = 1.0 + 0.06 * np.sin(np.pi * np.clip(t_exp / exp_dur, 0, 1))
            washout = 1.0 / (1.0 + np.exp((t_exp - (exp_dur - 0.16)) / 0.035))
            co2[idx:end] = np.where(t_loc <= ti, 0.0, co2_target * upstroke * dome * washout)

            pmus_proxy[idx:end] = 4.5
            idx = end

        # --- STATE 3: APNEA / EXTUBATION PAUSE ---
        elif state == 3:
            step = min(n_samples - idx, int(1.0 * fs))
            awp[idx:idx + step] = 0.0
            co2[idx:idx + step] = 0.0
            hypercapnia_offset = min(10.0, hypercapnia_offset + 0.35)
            idx += step

        # --- STATE 4: UNKNOWN / ARTIFACT (AWP Spikes + Flat 0 CO2) ---
        elif state == 4:
            step = min(n_samples - idx, int(rng.uniform(1.5, 3.0) * fs))
            t_loc = np.linspace(0, step / fs, step)
            spike = rng.uniform(10.0, 24.0) * np.exp(-((t_loc - 0.35) ** 2) / 0.025)
            awp[idx:idx + step] = spike
            co2[idx:idx + step] = 0.0  # Occluded sampling line / mask leak / esophageal
            idx += step

    # Apply realistic side-stream CO2 sampling lag & mild smoothing
    co2 = np.roll(co2, co2_lag_samples)
    co2[:co2_lag_samples] = 0.0
    co2 = gaussian_filter1d(co2, sigma=3)

    # 3. Simulate Sample-and-Hold Dynamic Compliance (updated every 8 seconds)
    cdyn = np.zeros(n_samples)
    update_step = 8 * fs
    last_cdyn = C_rs * 1000.0
    for start in range(0, n_samples, update_step):
        end = min(n_samples, start + update_step)
        delta_p = np.max(awp[start:end]) - np.min(awp[start:end])
        mean_vt = np.mean(vt_proxy[start:end])
        if delta_p > 5.0 and mean_vt > 100.0:
            # Drops during irregular transition/bagging or stabilizes during VCV
            effort_factor = 0.65 if np.mean(pmus_proxy[start:end]) > 1.0 else 1.0
            last_cdyn = np.clip((mean_vt / delta_p) * effort_factor, 14.0, 95.0)
        cdyn[start:end] = last_cdyn

    # Add realistic baseline sensor noise
    awp += rng.normal(0, 0.25, n_samples)
    co2 = np.maximum(0.0, co2 + rng.normal(0, 0.2, n_samples))

    return t, co2, awp, cdyn, labels


def plot_sample_gallery(seeds=(12, 42, 108, 256), duration_sec=480, fs=128):
    """
    Generates and plots 4 different synthetic anesthesia emergence cases
    in a 2x2 grid (CO2 + AWP + Regime Shading) plus a bottom row zooming
    specifically into each case's Transition window.
    """
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#121212")

    # Outer grid: 2 columns x 3 rows
    # Rows 0 & 1: Full-case overview (CO2 on top, AWP on bottom) for 4 seeds
    # Row 2: Zoomed-in Transition top-plateau morphology for each of the 4 seeds
    outer_gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], hspace=0.35, wspace=0.14)

    for idx, seed in enumerate(seeds[:4]):
        t, co2, awp, cdyn, labels = generate_synthetic_case(
            duration_sec=duration_sec, fs=fs, seed=seed
        )

        # Place first 4 cases in the top 2x2 grid
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
        ax_co2.set_title(f"Sample Seed #{seed} — Full Trajectory Overview", fontsize=10, color="#eeeeee", pad=4)
        plt.setp(ax_co2.get_xticklabels(), visible=False)

        ax_awp.axhline(0, color="#888888", linestyle="--", linewidth=0.7)
        ax_awp.plot(t, awp, color="#f5f5f5", linewidth=0.85)
        ax_awp.set_ylabel("AWP", fontsize=8.5, color="#cccccc")
        ax_awp.set_xlabel("Time (s)", fontsize=8.5, color="#aaaaaa")

        # Shade contiguous regimes
        change_idx = np.where(np.diff(labels) != 0)[0] + 1
        seg_starts = np.concatenate(([0], change_idx))
        seg_ends = np.concatenate((change_idx, [len(labels) - 1]))

        for s, e in zip(seg_starts, seg_ends):
            color = REGIME_COLORS[labels[s]]
            ax_awp.axvspan(t[s], t[e], color=color, alpha=0.45)
            ax_co2.axvspan(t[s], t[e], color=color, alpha=0.15)

    # Bottom Row: Side-by-side zoomed inspection of Transition CO2 top plateaus across all 4 seeds
    zoom_gs = outer_gs[2, :].subgridspec(1, 4, wspace=0.14)
    for idx, seed in enumerate(seeds[:4]):
        t, co2, awp, _, labels = generate_synthetic_case(
            duration_sec=duration_sec, fs=fs, seed=seed
        )
        trans_indices = np.where(labels == 1)[0]
        ax_zoom = fig.add_subplot(zoom_gs[0, idx])
        ax_zoom.set_facecolor("#181818")
        ax_zoom.grid(True, color="#2a2a2a", linewidth=0.5)

        if len(trans_indices) > 0:
            # Zoom into a 35-second slice inside the first Transition regime
            start_i = trans_indices[0]
            end_i = min(len(t) - 1, start_i + int(35 * fs))
            ax_zoom.plot(t[start_i:end_i], co2[start_i:end_i], color="#29b6f6", linewidth=1.3)
            ax_zoom.axvspan(t[start_i], t[end_i], color=REGIME_COLORS[1], alpha=0.2)
            ax_zoom.set_ylim(-2, 58)
            ax_zoom.set_title(f"Seed #{seed}: Transition CO2 Zoom (35s)", fontsize=9, color="#ffd54f")
            ax_zoom.set_xlabel("Time (s)", fontsize=8, color="#aaaaaa")
            if idx == 0:
                ax_zoom.set_ylabel("CO2 (mmHg)", fontsize=8.5, color="#cccccc")

        ax_zoom.spines["top"].set_visible(False)
        ax_zoom.spines["right"].set_visible(False)
        ax_zoom.spines["left"].set_color("#444444")
        ax_zoom.spines["bottom"].set_color("#444444")
        ax_zoom.tick_params(colors="#aaaaaa", labelsize=8)

    # Global Legend at the top
    patches = [
        mpatches.Patch(color=REGIME_COLORS[k], alpha=0.7, label=REGIME_NAMES[k])
        for k in sorted(REGIME_NAMES)
    ]
    fig.legend(
        handles=patches,
        loc="upper center",
        ncol=5,
        frameon=True,
        facecolor="#1e1e1e",
        edgecolor="#444444",
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.985),
    )

    plt.show()

if __name__ == "__main__":
    plot_sample_gallery(seeds=(12, 42, 108, 256))