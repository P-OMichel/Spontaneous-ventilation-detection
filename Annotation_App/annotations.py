import os
import json
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np


REGIME_CLASSES = [
    "Controlled",
    "Unknown",
    "Apnea",
    "Transition to Spontaneous ventilation",
    "Efficient spontaneous ventilation",
]

REGIME_TO_ID = {name: idx for idx, name in enumerate(REGIME_CLASSES)}
ID_TO_REGIME = {idx: name for idx, name in enumerate(REGIME_CLASSES)}

HYPNOTIC_AGENTS = ["unknown", "propofol", "sevoflurane", "mixed"]


@dataclass
class EventMarker:
    time: float
    event_type: str  # "spont_onset" or "extubation"


class AnnotationStore:
    """Holds ground-truth ventilation regime labels, crop boundaries, events, and selective .npz I/O."""

    def __init__(self, fs: int = 128, win_sec: float = 5.0, step_sec: float = 2.0):
        self.fs = fs
        self.win_sec = win_sec
        self.step_sec = step_sec

        self.sample_id: str = "uninitialized"
        self.source_path: str = ""
        self.hypnotic_type: str = "unknown"
        self.hypnotic_tracks: List[dict] = []

        # Raw / aligned 128 Hz waveforms
        self.t_common: np.ndarray = np.array([], dtype=np.float32)
        self.awp_data: np.ndarray = np.array([], dtype=np.float32)
        self.co2_data: np.ndarray = np.array([], dtype=np.float32)
        self.eeg_data: np.ndarray = np.array([], dtype=np.float32)
        self.ecg_data: np.ndarray = np.array([], dtype=np.float32)

        # Optional compliance track (lower sampling rate)
        self.compl_time: np.ndarray = np.array([], dtype=np.float32)
        self.compl_val: np.ndarray = np.array([], dtype=np.float32)

        # Windowed regime timeline (interactive step-level ground truth)
        self.t_windows: np.ndarray = np.array([], dtype=np.float32)
        self.regimes: np.ndarray = np.array([], dtype=np.int32)
        self.events: List[EventMarker] = []

        self.keep_start_sec: float = 0.0
        self.keep_end_sec: float = 0.0

        self.loaded_sections: dict = {
            "regimes": True,
            "events": True,
            "crop": True,
        }

    def initialize_record(
        self,
        sample_id: str,
        t_common: np.ndarray,
        awp_data: np.ndarray,
        co2_data: np.ndarray,
        eeg_data: np.ndarray,
        ecg_data: np.ndarray,
        compl_time: np.ndarray,
        compl_val: np.ndarray,
        t_windows: np.ndarray,
        initial_regimes: np.ndarray,
        source_path: str = "",
        hypnotic_type: str = "unknown",
        hypnotic_tracks: List[dict] | None = None,
    ):
        """Resets and populates the store for a newly loaded raw VitalDB recording."""
        self.sample_id = sample_id
        self.source_path = source_path
        self.hypnotic_type = hypnotic_type if hypnotic_type in HYPNOTIC_AGENTS else "unknown"
        self.hypnotic_tracks = hypnotic_tracks if hypnotic_tracks is not None else []

        self.t_common = t_common.astype(np.float32)
        self.awp_data = awp_data.astype(np.float32)
        self.co2_data = co2_data.astype(np.float32)
        self.eeg_data = eeg_data.astype(np.float32)
        self.ecg_data = ecg_data.astype(np.float32)
        self.compl_time = compl_time.astype(np.float32)
        self.compl_val = compl_val.astype(np.float32)

        self.t_windows = t_windows.astype(np.float32)
        self.regimes = initial_regimes.astype(np.int32)
        self.events.clear()

        self.keep_start_sec = float(self.t_common[0]) if len(self.t_common) > 0 else 0.0
        self.keep_end_sec = float(self.t_common[-1]) if len(self.t_common) > 0 else 0.0
        self.loaded_sections = {"regimes": True, "events": True, "crop": True}

    @staticmethod
    def is_annotated_npz(file_path: str) -> bool:
        """Returns True if the given .npz file is a ventilation-annotated archive."""
        if not file_path.lower().endswith(".npz") or not os.path.exists(file_path):
            return False
        try:
            with np.load(file_path, allow_pickle=False) as data:
                base_required = {"sample_id", "awp", "co2", "t_windows", "metadata_json"}
                return base_required.issubset(set(data.files))
        except Exception:
            return False

    @staticmethod
    def get_default_annotated_path(raw_file_path: str) -> str:
        """Returns the expected sidecar annotated filename in the exact same folder."""
        directory = os.path.dirname(raw_file_path)
        stem = os.path.splitext(os.path.basename(raw_file_path))[0]
        if stem.endswith("_vent_annotated"):
            return raw_file_path
        return os.path.join(directory, f"{stem}_vent_annotated.npz")

    def load_from_annotated_npz(
        self,
        annotated_path: str,
        fallback_initial_regimes_fn=None,
        hypnotic_tracks: List[dict] | None = None,
    ):
        """Restores waveforms and any selectively saved ventilation annotation sections."""
        with np.load(annotated_path, allow_pickle=False) as data:
            file_keys = set(data.files)
            meta = json.loads(str(data["metadata_json"]))
            saved_sections = meta.get("saved_sections", {})

            self.sample_id = str(data["sample_id"])
            self.source_path = annotated_path
            self.hypnotic_type = (
                str(data["hypnotic_type"])
                if "hypnotic_type" in file_keys
                else meta.get("hypnotic_type", "unknown")
            )
            self.hypnotic_tracks = hypnotic_tracks if hypnotic_tracks is not None else []

            self.t_common = data["t_common"].astype(np.float32)
            self.awp_data = data["awp"].astype(np.float32)
            self.co2_data = data["co2"].astype(np.float32)
            self.eeg_data = data["eeg"].astype(np.float32) if "eeg" in file_keys else np.zeros_like(self.awp_data)
            self.ecg_data = data["ecg"].astype(np.float32) if "ecg" in file_keys else np.zeros_like(self.awp_data)
            self.compl_time = data["compl_time"].astype(np.float32) if "compl_time" in file_keys else np.array([], dtype=np.float32)
            self.compl_val = data["compl_val"].astype(np.float32) if "compl_val" in file_keys else np.array([], dtype=np.float32)

            self.t_windows = data["t_windows"].astype(np.float32)

            # 1. Regime Labels
            has_regimes = saved_sections.get("regimes", "regime_labels" in file_keys) and (
                "regime_labels" in file_keys
            )
            if has_regimes:
                self.regimes = data["regime_labels"].astype(np.int32)
            elif fallback_initial_regimes_fn is not None:
                self.regimes = fallback_initial_regimes_fn(
                    self.t_common, self.awp_data, self.co2_data, self.t_windows
                ).astype(np.int32)
            else:
                self.regimes = np.full(len(self.t_windows), REGIME_TO_ID["Unknown"], dtype=np.int32)

            # 2. Crop Boundaries
            min_t = float(self.t_common[0]) if len(self.t_common) > 0 else 0.0
            max_t = float(self.t_common[-1]) if len(self.t_common) > 0 else 0.0
            has_crop = saved_sections.get("crop", "crop_start_sec" in meta)
            if has_crop:
                self.keep_start_sec = float(meta.get("crop_start_sec", min_t))
                self.keep_end_sec = float(meta.get("crop_end_sec", max_t))
            else:
                self.keep_start_sec = min_t
                self.keep_end_sec = max_t

            # 3. Transition Events
            has_events = saved_sections.get(
                "events",
                ("spont_onset_times" in file_keys) or ("extubation_times" in file_keys),
            )
            self.events.clear()
            if has_events:
                for t_sp in data.get("spont_onset_times", []):
                    self.events.append(EventMarker(time=float(t_sp), event_type="spont_onset"))
                for t_ex in data.get("extubation_times", []):
                    self.events.append(EventMarker(time=float(t_ex), event_type="extubation"))

            self.loaded_sections = {
                "regimes": bool(has_regimes),
                "events": bool(has_events),
                "crop": bool(has_crop),
            }

    def set_crop_bounds(self, start_sec: float, end_sec: float):
        min_t = float(self.t_common[0]) if len(self.t_common) > 0 else 0.0
        max_t = float(self.t_common[-1]) if len(self.t_common) > 0 else 0.0
        self.keep_start_sec = max(min_t, min(float(start_sec), max_t))
        self.keep_end_sec = max(self.keep_start_sec, min(float(end_sec), max_t))

    def set_regimes_at_indices(self, indices: Sequence[int], regime_idx: int):
        if regime_idx not in ID_TO_REGIME:
            raise ValueError(f"Invalid regime index: {regime_idx}")
        for idx in indices:
            if 0 <= idx < len(self.regimes):
                self.regimes[idx] = regime_idx

    def add_event(self, time_sec: float, event_type: str) -> EventMarker:
        if event_type not in ("spont_onset", "extubation"):
            raise ValueError(f"Unsupported event type: {event_type}")
        marker = EventMarker(time=float(time_sec), event_type=event_type)
        self.events.append(marker)
        return marker

    def update_event_time(self, index: int, new_time_sec: float):
        if 0 <= index < len(self.events):
            self.events[index].time = float(new_time_sec)

    def remove_event(self, index: int):
        if 0 <= index < len(self.events):
            self.events.pop(index)

    def get_contiguous_intervals(self) -> List[tuple[float, float, int]]:
        """Converts per-window integer regime labels into merged (t_start, t_end, regime_id) spans."""
        if len(self.t_windows) == 0:
            return []
        half_step = 0.5 * self.step_sec
        intervals = []
        cur_id = int(self.regimes[0])
        seg_start = float(self.t_windows[0]) - half_step

        for i in range(1, len(self.t_windows)):
            rid = int(self.regimes[i])
            if rid != cur_id:
                seg_end = 0.5 * (float(self.t_windows[i - 1]) + float(self.t_windows[i]))
                intervals.append((seg_start, seg_end, cur_id))
                seg_start = seg_end
                cur_id = rid

        seg_end = float(self.t_windows[-1]) + half_step
        intervals.append((seg_start, seg_end, cur_id))
        return intervals

    def save_annotated_npz(
        self,
        save_path: str,
        save_regimes: bool = True,
        save_events: bool = True,
        save_crop: bool = True,
    ) -> dict:
        """Selectively saves chosen ventilation annotation tracks to compressed .npz."""
        min_t = float(self.t_common[0]) if len(self.t_common) > 0 else 0.0
        max_t = float(self.t_common[-1]) if len(self.t_common) > 0 else 0.0

        effective_start = self.keep_start_sec if save_crop else min_t
        effective_end = self.keep_end_sec if save_crop else max_t

        valid_win_mask = (self.t_windows >= effective_start) & (self.t_windows <= effective_end)

        saved_sections = {
            "regimes": bool(save_regimes),
            "events": bool(save_events),
            "crop": bool(save_crop),
        }

        metadata = {
            "sample_id": self.sample_id,
            "hypnotic_type": self.hypnotic_type,
            "source_file": os.path.basename(self.source_path) if self.source_path else "",
            "sampling_rate_hz": self.fs,
            "window_size_sec": self.win_sec,
            "step_size_sec": self.step_sec,
            "crop_start_sec": round(effective_start, 3),
            "crop_end_sec": round(effective_end, 3),
            "regime_mapping": REGIME_TO_ID,
            "saved_sections": saved_sections,
        }

        payload = {
            "sample_id": np.array(self.sample_id),
            "hypnotic_type": np.array(self.hypnotic_type),
            "t_common": self.t_common,
            "awp": self.awp_data,
            "co2": self.co2_data,
            "eeg": self.eeg_data,
            "ecg": self.ecg_data,
            "compl_time": self.compl_time,
            "compl_val": self.compl_val,
            "t_windows": self.t_windows,
            "valid_window_mask": valid_win_mask,
            "metadata_json": np.array(json.dumps(metadata)),
        }

        if save_regimes:
            payload["regime_labels"] = self.regimes

        kept_events_count = 0
        if save_events:
            spont_times = np.array(
                [ev.time for ev in self.events if ev.event_type == "spont_onset"],
                dtype=np.float32,
            )
            extub_times = np.array(
                [ev.time for ev in self.events if ev.event_type == "extubation"],
                dtype=np.float32,
            )
            payload["spont_onset_times"] = spont_times
            payload["extubation_times"] = extub_times
            kept_events_count = sum(
                1 for ev in self.events if effective_start <= ev.time <= effective_end
            )

        for idx, trk in enumerate(self.hypnotic_tracks):
            payload[f"hypnotic_{idx}_time"] = trk["time"]
            payload[f"hypnotic_{idx}_val"] = trk["val"]
            payload[f"hypnotic_{idx}_name"] = np.array(trk["name"])

        np.savez_compressed(save_path, **payload)
        self.loaded_sections = saved_sections

        sample_mask = (self.t_common >= effective_start) & (self.t_common <= effective_end)
        return {
            "kept_samples": int(np.sum(sample_mask)),
            "kept_duration_sec": max(0.0, effective_end - effective_start),
            "kept_windows": int(np.sum(valid_win_mask)),
            "kept_events": kept_events_count,
            "saved_sections": saved_sections,
        }

    @staticmethod
    def load_cropped_for_ml(annotated_path: str) -> dict:
        """Loads a ventilation-annotated .npz file cropped and aligned for supervised ML."""
        with np.load(annotated_path, allow_pickle=False) as data:
            file_keys = set(data.files)
            meta = json.loads(str(data["metadata_json"]))
            crop_start = float(meta["crop_start_sec"])
            crop_end = float(meta["crop_end_sec"])

            t_common = data["t_common"]
            s_mask = (t_common >= crop_start) & (t_common <= crop_end)

            win_mask = data["valid_window_mask"].astype(bool)
            t_win_cropped = data["t_windows"][win_mask] - crop_start
            regimes_cropped = data["regime_labels"][win_mask] if "regime_labels" in file_keys else None

            spont_cropped = None
            extub_cropped = None
            if "spont_onset_times" in file_keys:
                sp = data["spont_onset_times"]
                spont_cropped = sp[(sp >= crop_start) & (sp <= crop_end)] - crop_start
            if "extubation_times" in file_keys:
                ex = data["extubation_times"]
                extub_cropped = ex[(ex >= crop_start) & (ex <= crop_end)] - crop_start

            return {
                "sample_id": str(data["sample_id"]),
                "hypnotic_type": str(data["hypnotic_type"]),
                "t": t_common[s_mask] - crop_start,
                "awp": data["awp"][s_mask],
                "co2": data["co2"][s_mask],
                "eeg": data["eeg"][s_mask] if "eeg" in file_keys else None,
                "ecg": data["ecg"][s_mask] if "ecg" in file_keys else None,
                "t_windows": t_win_cropped,
                "regime_labels": regimes_cropped,
                "spont_onset_times": spont_cropped,
                "extubation_times": extub_cropped,
                "metadata": meta,
            }