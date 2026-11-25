import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ProcessPoolExecutor, as_completed

REGIME_NAMES = {
    0: "Controlled",
    1: "Transition",
    2: "Efficient Spontaneous",
    3: "Apnea",
    4: "Unknown / Artifact",
}


def generate_synthetic_case(duration_sec=500, fs=128, seed=None):
    """
    Generates coupled synthetic anesthesia waveforms (CO2, AWP, Cdyn) at 128 Hz,
    including enriched Unknown / Artifact regimes (~12-18% prevalence) featuring:
      - Absurd out-of-bounds values (high-vacuum suctioning, over-pressure spikes, CO2 rail clipping)
      - Cross-channel physiological impossibilities:
          * Active VCV pressure waves with flat 0 CO2 (sampling line disconnect / bypass)
          * Active CO2 waveforms with flat 0 AWP (pressure line disconnect / transducer fault)
          * Uncoupled / chaotic surgical & electrocautery interference
          * Frozen non-zero sensor flatlines
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

    # 1. Build clinical trajectory templates with higher Artifact representation
    labels = np.zeros(n_samples, dtype=int)
    cursor = 0
    trajectory_templates = [
        # Template A: Artifact during Controlled phase + standard emergence
        [
            (0, rng.uniform(60, 90)),    # Controlled
            (4, rng.uniform(25, 45)),    # Unknown / Artifact (e.g., sampling line clog or suctioning)
            (0, rng.uniform(50, 80)),    # Controlled resumes
            (1, rng.uniform(45, 70)),    # Transition
            (2, rng.uniform(75, 115)),   # Efficient Spontaneous
            (3, rng.uniform(25, 45)),    # Extubation Apnea
        ],
        # Template B: Multi-artifact emergence (suctioning, transducer fault, rescue bagging)
        [
            (0, rng.uniform(85, 120)),   # Controlled
            (1, rng.uniform(40, 65)),    # Transition 1
            (4, rng.uniform(25, 45)),    # Unknown / Artifact 1
            (3, rng.uniform(15, 25)),    # Apnea pause
            (1, rng.uniform(35, 55)),    # Transition 2
            (2, rng.uniform(65, 100)),   # Efficient Spontaneous
            (4, rng.uniform(20, 35)),    # Unknown / Artifact 2 (e.g., open-air CO2 without AWP)
            (3, rng.uniform(25, 45)),    # Extubation Apnea
        ],
        # Template C: Early artifact + chaotic emergence + spontaneous recovery
        [
            (4, rng.uniform(20, 35)),    # Unknown / Artifact (induction / esophageal / sensor check)
            (0, rng.uniform(90, 130)),   # Controlled
            (1, rng.uniform(50, 80)),    # Transition
            (4, rng.uniform(25, 40)),    # Unknown / Artifact (suctioning / leak)
            (2, rng.uniform(80, 120)),   # Efficient Spontaneous
            (3, rng.uniform(25, 45)),    # Apnea
        ],
    ]
    sequence = trajectory_templates[rng.integers(0, len(trajectory_templates))]

    for state, dur in sequence:
        end = min(n_samples, cursor + int(dur * fs))
        labels[cursor:end] = state
        cursor = end
        if cursor >= n_samples:
            break

    awp = np.zeros(n_samples)
    co2 = np.zeros(n_samples)
    vt_proxy = np.zeros(n_samples)
    pmus_proxy = np.zeros(n_samples)

    # 2. Synthesize breath-by-breath waveforms per regime
    idx = 0
    hypercapnia_offset = 0.0

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

            awp[idx:end] = np.where(
                t_loc <= ti,
                peep_set + R_rs * flow_insp + (flow_insp * t_loc / C_rs),
                peep_set + (vt / C_rs) * np.exp(-(t_loc - ti) / (R_rs * C_rs)) * 0.4,
            )

            co2_target = alv_co2_base + hypercapnia_offset
            hypercapnia_offset = max(0.0, hypercapnia_offset - 1.5)

            t_exp = np.maximum(0.0, t_loc - ti)
            exp_dur = max(0.2, dur_s - ti)
            upstroke = 1.0 - np.exp(-t_exp / 0.14)
            washout = 1.0 / (1.0 + np.exp((t_exp - (exp_dur - 0.18)) / 0.03))
            co2[idx:end] = np.where(t_loc <= ti, 0.0, co2_target * upstroke * washout)

            vt_proxy[idx:end] = vt * 1000.0
            idx = end

        # --- STATE 1: TRANSITION (Wide Plateaus, Clefts, Stacking & Bagging) ---
        elif state == 1:
            sub_pattern = rng.choice(
                ["wide_plateau", "curare_cleft", "stacked_cluster", "manual_bagging"],
                p=[0.30, 0.30, 0.25, 0.15],
            )

            if sub_pattern == "wide_plateau":
                dur_s = rng.uniform(4.5, 8.5)
                end = min(n_samples, idx + int(dur_s * fs))
                t_loc = np.linspace(0, (end - idx) / fs, end - idx)
                ti = rng.uniform(0.45, 0.75)

                awp[idx:end] = np.where(
                    t_loc <= ti,
                    -rng.uniform(2.5, 5.5) * np.sin(np.pi * t_loc / ti),
                    rng.uniform(0.0, 1.5),
                )

                t_exp = np.maximum(0.0, t_loc - ti)
                exp_dur = max(0.5, (end - idx) / fs - ti)
                upstroke = 1.0 - np.exp(-t_exp / 0.18)
                plateau_ramp = (alv_co2_base + hypercapnia_offset) + rng.uniform(4.0, 9.0) * (t_exp / exp_dur)
                washout = 1.0 / (1.0 + np.exp((t_exp - (exp_dur - 0.15)) / 0.03))
                co2[idx:end] = np.where(t_loc <= ti, 0.0, plateau_ramp * upstroke * washout)

                hypercapnia_offset = min(10.0, hypercapnia_offset + 2.5)
                pmus_proxy[idx:end] = 4.0
                idx = end

            elif sub_pattern == "curare_cleft":
                dur_s = rng.uniform(3.8, 6.5)
                end = min(n_samples, idx + int(dur_s * fs))
                t_loc = np.linspace(0, (end - idx) / fs, end - idx)
                ti = rng.uniform(0.45, 0.70)

                t_exp = np.maximum(0.0, t_loc - ti)
                exp_dur = max(0.5, (end - idx) / fs - ti)
                upstroke = 1.0 - np.exp(-t_exp / 0.15)
                washout = 1.0 / (1.0 + np.exp((t_exp - (exp_dur - 0.15)) / 0.03))
                base_top = alv_co2_base + hypercapnia_offset + rng.uniform(1.0, 5.0)

                cleft_factor = np.ones_like(t_loc)
                awp_cleft = np.zeros_like(t_loc)
                n_clefts = rng.choice([1, 2], p=[0.7, 0.3])
                for c in range(n_clefts):
                    c_center = ti + exp_dur * rng.uniform(0.28 + 0.3 * c, 0.50 + 0.25 * c)
                    c_width = rng.uniform(0.14, 0.26)
                    c_depth = rng.uniform(0.45, 0.85)
                    notch = c_depth * np.exp(-((t_loc - c_center) ** 2) / (2 * c_width ** 2))
                    cleft_factor -= notch
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
                n_breaths = rng.integers(3, 6)
                for b in range(n_breaths):
                    if idx >= n_samples or labels[idx] != 1:
                        break
                    b_dur = rng.uniform(1.4, 2.5)
                    end = min(n_samples, idx + int(b_dur * fs))
                    t_loc = np.linspace(0, (end - idx) / fs, end - idx)
                    ti = 0.35 * b_dur

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
                n_squeezes = rng.integers(2, 5)
                for _ in range(n_squeezes):
                    if idx >= n_samples or labels[idx] != 1:
                        break
                    b_dur = rng.uniform(2.0, 3.4)
                    end = min(n_samples, idx + int(b_dur * fs))
                    t_loc = np.linspace(0, (end - idx) / fs, end - idx)
                    ti = rng.uniform(0.5, 0.8)

                    bag_peak = rng.uniform(22.0, 42.0)
                    p_bag = bag_peak * np.exp(-((t_loc - ti) ** 2) / 0.035)
                    p_dip = rng.uniform(2.5, 5.5) * np.exp(-((t_loc - ti * 0.4) ** 2) / 0.01)
                    awp[idx:end] = rng.uniform(0.5, 3.0) + p_bag - p_dip

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
            rr = rng.uniform(15.0, 20.0)
            cycle_len = int((60.0 / rr) * rng.normal(1.0, 0.06) * fs)
            end = min(n_samples, idx + max(int(1.6 * fs), cycle_len))
            dur_s = (end - idx) / fs
            t_loc = np.linspace(0, dur_s, end - idx)
            ti = 0.38 * dur_s

            amp_awp = rng.uniform(2.0, 3.5)
            awp[idx:end] = np.where(
                t_loc <= ti,
                -amp_awp * np.sin(np.pi * t_loc / ti),
                (amp_awp * 0.85) * np.sin(np.pi * (t_loc - ti) / (dur_s - ti)),
            )

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

        # --- STATE 4: UNKNOWN / ARTIFACT (Absurd Values & Physiological Contradictions) ---
        elif state == 4:
            # Find the full contiguous length of this Artifact block so one coherent fault occurs
            block_end = idx
            while block_end < n_samples and labels[block_end] == 4:
                block_end += 1
            block_len = block_end - idx
            dur_s = block_len / fs
            t_loc = np.linspace(0, dur_s, block_len)

            artifact_type = rng.choice([
                "paw_active_co2_zero",     # Mechanical pressure waves with flat 0 CO2 (sampling clog / esophageal)
                "co2_active_paw_zero",     # Normal CO2 waves with flat 0 AWP (transducer disconnect / room air)
                "absurd_paw_spikes",       # Over-pressure / high-vacuum suctioning (+60 to +105 or -25 hPa)
                "absurd_co2_rail",         # Capnograph sensor fault pinned at absurd 85-125 mmHg
                "uncoupled_cpr_cautery",   # High-frequency / anti-phase oscillations impossible physiologically
                "frozen_non_zero",         # Flatlined non-zero stuck sensor values
            ])

            if artifact_type == "paw_active_co2_zero":
                # Regular VCV-like pressure cycles (15-24 hPa), but CO2 is completely dead at 0 mmHg
                freq = rng.uniform(0.2, 0.28)
                phase = (t_loc * freq) % 1.0
                awp[idx:block_end] = np.where(
                    phase < 0.35,
                    peep_set + rng.uniform(16.0, 24.0) * (phase / 0.35),
                    peep_set + 3.0 * np.exp(-(phase - 0.35) * 6.0),
                )
                co2[idx:block_end] = 0.0

            elif artifact_type == "co2_active_paw_zero":
                # Normal-looking CO2 waves (35-44 mmHg), but AWP is completely flat at 0.0 hPa
                freq = rng.uniform(0.22, 0.32)
                phase = (t_loc * freq) % 1.0
                awp[idx:block_end] = 0.0
                co2[idx:block_end] = np.where(
                    phase < 0.35,
                    0.0,
                    alv_co2_base * (1.0 - np.exp(-(phase - 0.35) * 14.0)),
                )

            elif artifact_type == "absurd_paw_spikes":
                # Extreme non-physiological pressures: over-pressurization (+60..+105 hPa) or suction (-15..-30 hPa)
                awp[idx:block_end] = rng.uniform(0.0, 2.0)
                n_events = max(2, int(dur_s / 3.0))
                for _ in range(n_events):
                    center = rng.uniform(0.5, max(0.6, dur_s - 0.5))
                    width = rng.uniform(0.15, 0.45)
                    if rng.random() < 0.65:
                        amp = rng.uniform(60.0, 105.0)   # Absurd positive pressure spike
                    else:
                        amp = -rng.uniform(15.0, 30.0)   # Deep endotracheal suction vacuum
                    awp[idx:block_end] += amp * np.exp(-((t_loc - center) ** 2) / (2 * width ** 2))
                co2[idx:block_end] = 0.0

            elif artifact_type == "absurd_co2_rail":
                # Capnograph rails at absurdly high 85-125 mmHg while AWP shows random/normal ripples
                co2[idx:block_end] = rng.uniform(85.0, 125.0) + rng.normal(0, 1.5, block_len)
                awp[idx:block_end] = rng.uniform(2.0, 8.0) * np.sin(2 * np.pi * 0.25 * t_loc)

            elif artifact_type == "uncoupled_cpr_cautery":
                # Rapid chest compressions / surgical compression (1.5 - 2.2 Hz) uncoupled from CO2
                cpr_freq = rng.uniform(1.5, 2.2)
                awp[idx:block_end] = rng.uniform(12.0, 28.0) * np.maximum(0.0, np.sin(2 * np.pi * cpr_freq * t_loc))
                # Low, erratic pseudo-CO2 ripples at an unrelated slow frequency
                co2[idx:block_end] = rng.uniform(6.0, 14.0) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t_loc))

            else:  # "frozen_non_zero"
                # Stuck sensor line holding impossible non-zero static values
                awp[idx:block_end] = rng.uniform(14.0, 28.0)
                co2[idx:block_end] = rng.uniform(25.0, 42.0)

            idx = block_end

    # Apply realistic side-stream CO2 sampling lag & mild smoothing (preserve sharp artifacts)
    co2_shifted = np.roll(co2, co2_lag_samples)
    co2_shifted[:co2_lag_samples] = 0.0
    # Keep Artifact blocks unlagged/unblurred where CO2 is explicitly forced to 0 or railed
    artifact_mask = labels == 4
    co2 = np.where(artifact_mask, co2, gaussian_filter1d(co2_shifted, sigma=3))

    # 3. Simulate Sample-and-Hold Dynamic Compliance (updated every 8 seconds)
    cdyn = np.zeros(n_samples)
    update_step = 8 * fs
    last_cdyn = C_rs * 1000.0
    for start in range(0, n_samples, update_step):
        end = min(n_samples, start + update_step)
        # During artifact blocks, Cdyn either freezes or jumps to absurd values (>150 or <5)
        if np.mean(labels[start:end] == 4) > 0.5:
            if rng.random() < 0.4:
                last_cdyn = rng.choice([rng.uniform(2.0, 6.0), rng.uniform(160.0, 240.0)])
        else:
            delta_p = np.max(awp[start:end]) - np.min(awp[start:end])
            mean_vt = np.mean(vt_proxy[start:end])
            if delta_p > 5.0 and mean_vt > 100.0:
                effort_factor = 0.65 if np.mean(pmus_proxy[start:end]) > 1.0 else 1.0
                last_cdyn = np.clip((mean_vt / delta_p) * effort_factor, 14.0, 95.0)
        cdyn[start:end] = last_cdyn

    # Add baseline sensor noise (except on frozen sensor artifacts)
    frozen_mask = (np.gradient(awp) == 0) & (labels == 4)
    awp = np.where(frozen_mask, awp, awp + rng.normal(0, 0.25, n_samples))
    co2 = np.where(frozen_mask, co2, np.maximum(0.0, co2 + rng.normal(0, 0.2, n_samples)))

    return t, co2, awp, cdyn, labels


def _generate_single_worker(args):
    case_idx, seed, duration_sec, fs = args
    _, co2, awp, cdyn, labels = generate_synthetic_case(
        duration_sec=duration_sec, fs=fs, seed=seed
    )
    signals = np.stack([co2, awp, cdyn], axis=0).astype(np.float32)
    return case_idx, signals, labels.astype(np.int64)


def build_and_save_dataset(
    output_dir="anesthesia_synth_dataset",
    n_train=800,
    n_val=100,
    n_test=100,
    duration_sec=500,
    fs=128,
    base_seed=2026,
    n_workers=None,
):
    os.makedirs(output_dir, exist_ok=True)
    n_samples = int(duration_sec * fs)

    splits = {
        "train": (n_train, base_seed),
        "val":   (n_val,   base_seed + 100_000),
        "test":  (n_test,  base_seed + 200_000),
    }

    metadata = {
        "fs": fs,
        "duration_sec": duration_sec,
        "n_samples_per_case": n_samples,
        "channels": ["CO2_mmHg", "AWP_hPa", "Cdyn_mL_mbar"],
        "regimes": {str(k): v for k, v in REGIME_NAMES.items()},
        "splits": {},
    }

    for split_name, (n_cases, seed_offset) in splits.items():
        print(f"\nGenerating '{split_name}' split ({n_cases} cases, {duration_sec}s @ {fs}Hz)...")
        X = np.zeros((n_cases, 3, n_samples), dtype=np.float32)
        y = np.zeros((n_cases, n_samples), dtype=np.int64)

        tasks = [(i, seed_offset + i, duration_sec, fs) for i in range(n_cases)]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_generate_single_worker, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), 1):
                case_idx, signals, labels = future.result()
                X[case_idx] = signals
                y[case_idx] = labels
                if completed % max(1, n_cases // 5) == 0 or completed == n_cases:
                    print(f"  [{split_name}] {completed}/{n_cases} cases complete")

        unique, counts = np.unique(y, return_counts=True)
        total_pts = y.size
        class_dist = {
            REGIME_NAMES[int(u)]: round(float(c / total_pts) * 100, 2)
            for u, c in zip(unique, counts)
        }
        print(f"  [{split_name}] Regime distribution (%): {class_dist}")

        out_path = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(out_path, X=X, y=y)
        file_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  Saved -> {out_path} ({file_mb:.1f} MB)")

        metadata["splits"][split_name] = {
            "n_cases": n_cases,
            "seed_range": [seed_offset, seed_offset + n_cases - 1],
            "class_distribution_pct": class_dist,
            "file": f"{split_name}.npz",
        }

    train_data = np.load(os.path.join(output_dir, "train.npz"))["X"]
    channel_means = train_data.mean(axis=(0, 2)).tolist()
    channel_stds = train_data.std(axis=(0, 2)).tolist()
    metadata["train_normalization"] = {
        "mean": dict(zip(metadata["channels"], [round(m, 4) for m in channel_means])),
        "std": dict(zip(metadata["channels"], [round(s, 4) for s in channel_stds])),
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDataset build complete! Metadata saved to {meta_path}")


if __name__ == "__main__":
    build_and_save_dataset(
        output_dir="anesthesia_synth_dataset",
        n_train=1000,
        n_val=200,
        n_test=200,
        duration_sec=500,
        fs=128,
    )