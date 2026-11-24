import os
import numpy as np
from scipy import signal

from Annotation_App.annotations import REGIME_TO_ID


class VentilationProcessor:
    """Extracts VitalDB respiratory/EEG/hypnotic tracks (last 1h max) and computes initial regime estimates."""

    PROPOFOL_TRACKS = {
        "Orchestra__PPF20_CE": "Propofol Ce (µg/mL)",
        "Orchestra__PPF20_CP": "Propofol Cp (µg/mL)",
        "Orchestra__PPF_CE": "Propofol Ce (µg/mL)",
        "Orchestra__PPF_CP": "Propofol Cp (µg/mL)",
    }

    SEVO_TRACKS = {
        "Primus__EXP_SEVO": "Sevoflurane Exp (%)",
        "Solar8000__ETSEVO": "Sevoflurane ET (%)",
        "Primus__INSP_SEVO": "Sevoflurane Insp (%)",
        "Solar8000__INSEVO": "Sevoflurane Insp (%)",
    }

    def __init__(self, fs: int = 128, max_duration_sec: float = 3600.0):
        self.fs = fs
        self.max_duration_sec = max_duration_sec

    @staticmethod
    def _get_first_track(
        data: np.lib.npyio.NpzFile, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        files = set(data.files)
        for key in candidates:
            t_key, v_key = f"{key}_time", f"{key}_val"
            if t_key in files and v_key in files:
                t = np.asarray(data[t_key], dtype=np.float64)
                v = np.asarray(data[v_key], dtype=np.float64)

                outliers = np.isfinite(v) & (np.abs(v) > 1e5)
                if np.any(outliers):
                    v = v.copy()
                    v[outliers] = np.nan

                finite = np.isfinite(t) & np.isfinite(v)
                if np.any(finite):
                    return t[np.isfinite(t)], v[np.isfinite(t)]
        return None

    def extract_signals_from_file(self, file_path: str) -> dict:
        """
        Loads and aligns up to the last `max_duration_sec` (default 3600s = 1h)
        of 128 Hz AWP, CO2, EEG, ECG, plus Compliance from a VitalDB .npz file.
        """
        with np.load(file_path, allow_pickle=True) as data:
            awp_raw = self._get_first_track(data, ["Primus__AWP"])
            co2_raw = self._get_first_track(data, ["Primus__CO2"])
            if awp_raw is None or co2_raw is None:
                raise KeyError(
                    f"Both Primus__AWP and Primus__CO2 are required in {os.path.basename(file_path)}."
                )

            eeg_raw = self._get_first_track(data, ["BIS__EEG1_WAV", "BIS__EEG2_WAV"])
            ecg_raw = self._get_first_track(data, ["SNUADC__ECG_II", "SNUADC__ECG_V5"])
            compl_raw = self._get_first_track(
                data, ["Primus__COMPLIANCE", "Solar8000__VENT_COMPL"]
            )

        t_end = min(float(awp_raw[0][-1]), float(co2_raw[0][-1]))
        t_start = max(float(awp_raw[0][0]), float(co2_raw[0][0]))

        # Restrict to at most the last 1 hour (3600s)
        if self.max_duration_sec > 0:
            t_start = max(t_start, t_end - float(self.max_duration_sec))

        if t_end <= t_start + 5.0:
            raise ValueError("Overlapping AWP/CO2 duration is too short (< 5s).")

        dt = 1.0 / float(self.fs)
        t_common = np.arange(t_start, t_end, dt, dtype=np.float32)

        def _resample_to_grid(track: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
            if track is None:
                return np.zeros(len(t_common), dtype=np.float32)
            t_src, v_src = track
            valid = (
                np.isfinite(t_src)
                & np.isfinite(v_src)
                & (t_src >= t_start - 5.0)
                & (t_src <= t_end + 5.0)
            )
            if np.sum(valid) < 2:
                return np.zeros(len(t_common), dtype=np.float32)
            interp = np.interp(t_common, t_src[valid], v_src[valid], left=0.0, right=0.0)
            return interp.astype(np.float32)

        awp_aligned = _resample_to_grid(awp_raw)
        co2_aligned = _resample_to_grid(co2_raw)
        eeg_aligned = _resample_to_grid(eeg_raw)
        ecg_aligned = _resample_to_grid(ecg_raw)

        if compl_raw is not None:
            valid_cp = (
                np.isfinite(compl_raw[0])
                & np.isfinite(compl_raw[1])
                & (compl_raw[0] >= t_start)
                & (compl_raw[0] <= t_end)
            )
            compl_t = compl_raw[0][valid_cp].astype(np.float32)
            compl_v = compl_raw[1][valid_cp].astype(np.float32)
        else:
            compl_t = np.array([], dtype=np.float32)
            compl_v = np.array([], dtype=np.float32)

        return {
            "t_common": t_common,
            "awp": awp_aligned,
            "co2": co2_aligned,
            "eeg": eeg_aligned,
            "ecg": ecg_aligned,
            "compl_time": compl_t,
            "compl_val": compl_v,
        }

    def extract_hypnotics_from_file(self, file_path: str) -> tuple[list[dict], str]:
        tracks = []
        if not file_path.lower().endswith(".npz") or not os.path.exists(file_path):
            return tracks, "unknown"

        has_propofol = False
        has_sevo = False

        with np.load(file_path, allow_pickle=True) as data:
            file_keys = set(data.files)

            t_cutoff = -np.inf
            if self.max_duration_sec > 0:
                if "t_common" in file_keys and len(data["t_common"]) > 0:
                    t_cutoff = float(data["t_common"][-1]) - float(self.max_duration_sec)
                elif "Primus__AWP_time" in file_keys and len(data["Primus__AWP_time"]) > 0:
                    awp_t = np.asarray(data["Primus__AWP_time"], dtype=np.float64)
                    fin_t = awp_t[np.isfinite(awp_t)]
                    if len(fin_t) > 0:
                        t_cutoff = float(fin_t[-1]) - float(self.max_duration_sec)

            for base_key, display_name in self.PROPOFOL_TRACKS.items():
                val_key, time_key = f"{base_key}_val", f"{base_key}_time"
                if val_key in file_keys and time_key in file_keys:
                    vals = np.asarray(data[val_key], dtype=np.float32)
                    times = np.asarray(data[time_key], dtype=np.float32)
                    valid = np.isfinite(vals) & np.isfinite(times) & (times >= t_cutoff)
                    if np.any(valid) and np.nanmax(vals[valid]) > 0.0:
                        has_propofol = True
                        tracks.append({
                            "name": display_name,
                            "agent": "propofol",
                            "time": times[valid],
                            "val": vals[valid],
                        })

            for base_key, display_name in self.SEVO_TRACKS.items():
                val_key, time_key = f"{base_key}_val", f"{base_key}_time"
                if val_key in file_keys and time_key in file_keys:
                    vals = np.asarray(data[val_key], dtype=np.float32)
                    times = np.asarray(data[time_key], dtype=np.float32)
                    valid = np.isfinite(vals) & np.isfinite(times) & (times >= t_cutoff)
                    if np.any(valid) and np.nanmax(vals[valid]) > 0.0:
                        has_sevo = True
                        tracks.append({
                            "name": display_name,
                            "agent": "sevoflurane",
                            "time": times[valid],
                            "val": vals[valid],
                        })

        if has_propofol and has_sevo:
            detected_agent = "mixed"
        elif has_propofol:
            detected_agent = "propofol"
        elif has_sevo:
            detected_agent = "sevoflurane"
        else:
            detected_agent = "unknown"

        return tracks, detected_agent

    def filter_eeg(self, eeg: np.ndarray, low_hz: float = 0.5, high_hz: float = 45.0) -> np.ndarray:
        if len(eeg) <= int(self.fs * 3):
            return eeg.copy()
        nyq = 0.5 * self.fs
        sos = signal.butter(
            4, [low_hz / nyq, min(high_hz, 0.45 * self.fs) / nyq], btype="band", output="sos"
        )
        return signal.sosfiltfilt(sos, eeg).astype(np.float32)

    def compute_spectrogram(self, eeg: np.ndarray):
        """Computes STFT spectrogram for display (excluded from saved .npz files)."""
        nperseg = self.fs * 2
        noverlap = int(self.fs * 1.5)
        f, t_spec, Sxx = signal.spectrogram(eeg, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
        freq_mask = (f >= 0.5) & (f <= 35.0)
        f = f[freq_mask]
        Sxx_db = 10.0 * np.log10(Sxx[freq_mask, :] + 1e-12)
        return f, t_spec, Sxx_db

    def compute_window_grid(
        self, t_common: np.ndarray, win_sec: float = 5.0, step_sec: float = 2.0
    ) -> np.ndarray:
        """Creates uniform sliding window center timestamps across the recording."""
        if len(t_common) == 0:
            return np.array([], dtype=np.float32)
        t0 = float(t_common[0]) + 0.5 * win_sec
        t1 = float(t_common[-1]) - 0.5 * win_sec
        if t1 <= t0:
            return np.array([0.5 * (float(t_common[0]) + float(t_common[-1]))], dtype=np.float32)
        return np.arange(t0, t1 + 1e-6, step_sec, dtype=np.float32)

    def estimate_initial_regimes(
        self,
        t_common: np.ndarray,
        awp: np.ndarray,
        co2: np.ndarray,
        t_windows: np.ndarray,
    ) -> np.ndarray:
        """
        Proposes initial integer regime IDs on `t_windows`:
          0: Controlled
          1: Unknown
          2: Apnea
          3: Transition to Spontaneous ventilation
          4: Efficient spontaneous ventilation
        """
        n_win = len(t_windows)
        if n_win == 0 or len(t_common) < int(self.fs * 5):
            return np.full(n_win, REGIME_TO_ID["Unknown"], dtype=np.int32)

        win_smooth = max(1, int(round(0.05 * self.fs)))
        v_p_smooth = np.convolve(awp, np.ones(win_smooth) / win_smooth, mode="same")

        # 1. Detect breath cycles using CO2 expiratory upstrokes (hysteresis 10 -> 4 mmHg)
        cycle_times = []
        armed = True
        last_t = -np.inf
        for idx in range(len(co2)):
            if armed and co2[idx] > 10.0 and (t_common[idx] - last_t) >= 1.2:
                cycle_times.append(float(t_common[idx]))
                last_t = float(t_common[idx])
                armed = False
            elif not armed and co2[idx] < 4.0:
                armed = True

        has_co2_track = len(cycle_times) >= 3
        if not has_co2_track:
            peaks, _ = signal.find_peaks(v_p_smooth, prominence=2.5, distance=int(1.5 * self.fs))
            cycle_times = [float(t_common[pk]) for pk in peaks]

        if len(cycle_times) < 2:
            return np.full(n_win, REGIME_TO_ID["Apnea"], dtype=np.int32)

        # 2. Extract full-cycle AWP features (unaffected by CO2 transit lag)
        records = []
        for i in range(1, len(cycle_times)):
            t0 = cycle_times[i - 1]
            t1 = cycle_times[i]
            dt = t1 - t0

            if dt > 12.0:
                records.append((t0, t1, REGIME_TO_ID["Apnea"], dt, None))
                continue

            i0 = np.searchsorted(t_common, t0)
            i1 = np.searchsorted(t_common, t1)
            if i1 - i0 < int(0.4 * self.fs):
                continue

            seg_p = v_p_smooth[i0:i1]
            seg_c = co2[i0:i1]

            # True expiratory baseline PEEP (15th percentile of full respiratory cycle)
            peep = float(np.percentile(seg_p, 15))
            p_peak = float(np.percentile(seg_p, 98))
            p_min = float(np.min(seg_p))
            delta_p = p_peak - peep          # Positive pressure support above PEEP
            neg_dip = peep - p_min           # Negative inspiratory effort below PEEP
            etco2 = float(np.percentile(seg_c, 95)) if has_co2_track else 35.0

            records.append((t0, t1, None, dt, (delta_p, neg_dip, p_min, etco2)))

        # 3. Classify each respiratory cycle
        segments = []
        for i, (t0, t1, preset_id, dt, feats) in enumerate(records):
            if preset_id is not None:
                segments.append((t0, t1, preset_id))
                continue

            delta_p, neg_dip, p_min, etco2 = feats

            # Local 5-breath timing & pressure variability
            lo, hi = max(0, i - 2), min(len(records), i + 3)
            local_valid = [records[j] for j in range(lo, hi) if records[j][2] is None]
            local_dts = [r[3] for r in local_valid]
            local_dps = [r[4][0] for r in local_valid]

            cv_dt = (
                float(np.std(local_dts) / np.mean(local_dts))
                if len(local_dts) >= 3 and np.mean(local_dts) > 0
                else 0.0
            )
            cv_dp = (
                float(np.std(local_dps) / np.mean(local_dps))
                if len(local_dps) >= 3 and np.mean(local_dps) > 1.0
                else 0.0
            )

            # A. Apnea / Disconnected: no meaningful pressure swing and low/absent CO2
            if delta_p < 1.5 and neg_dip < 0.25 and etco2 < 12.0:
                rid = REGIME_TO_ID["Apnea"]

            # B. Unassisted Spontaneous Ventilation:
            #    Low positive pressure swing (< 4.5 hPa) WITH negative inspiratory pull
            elif delta_p < 4.5 and (neg_dip > 0.25 or p_min < -0.25):
                if cv_dt <= 0.16 and 2.5 <= dt <= 7.5 and etco2 >= 25.0:
                    rid = REGIME_TO_ID["Efficient spontaneous ventilation"]
                else:
                    rid = REGIME_TO_ID["Transition to Spontaneous ventilation"]

            # C. Assisted / PSV / Bucking / Manual Bagging:
            #    Positive pressure support (>= 4.5 hPa) WITH patient trigger dip or irregular timing/pressure
            elif neg_dip > 0.45 or cv_dt > 0.10 or cv_dp > 0.14:
                rid = REGIME_TO_ID["Transition to Spontaneous ventilation"]

            # D. Controlled Mechanical Ventilation:
            #    Steady positive driving pressure (>= 4.5 hPa), regular rate, no trigger dip
            else:
                rid = REGIME_TO_ID["Controlled"]

            segments.append((t0, t1, rid))

        # 4. Project breath segments onto `t_windows`
        window_labels = np.full(n_win, REGIME_TO_ID["Unknown"], dtype=np.int32)
        for t0, t1, rid in segments:
            mask = (t_windows >= t0) & (t_windows <= t1)
            window_labels[mask] = rid

        if segments:
            if segments[0][0] - float(t_common[0]) > 12.0:
                window_labels[t_windows < segments[0][0]] = REGIME_TO_ID["Apnea"]
            if float(t_common[-1]) - segments[-1][1] > 12.0:
                window_labels[t_windows > segments[-1][1]] = REGIME_TO_ID["Apnea"]

        return window_labels