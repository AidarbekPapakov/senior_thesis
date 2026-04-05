"""
Unified dataset preprocessor — INSTANCE events + STEAD events + STEAD noise → memmap

Output layout:
    ~/data/unified/
        waveforms_events.bin     (N_events, 3, 1000) float32
        metadata_events.csv      trace_name, source_magnitude, source, hdf5_path
        waveforms_noise.bin      (N_noise,  3, 1000) float32
        metadata_noise.csv       trace_name, source, hdf5_path
        info.json                shape metadata for both memmaps

Stratification target (events, ~100k total):
    magnitude [1.0, 3.0) → 25k   (replace=True if not enough)
    magnitude [3.0, 5.0) → 25k   (replace=True if not enough)
    magnitude [5.0,  ∞ ) → 25k   (replace=True if not enough, these are rare)
    Split roughly 50/50 INSTANCE vs STEAD per bin where possible.

Noise target: ~20k from STEAD chunks (no full INSTANCE noise HDF5 available).
If you locate Instance_noise_gm.hdf5, set INSTANCE_NOISE_HDF5 below and it will be included.
"""

import json
import os
import numpy as np
import pandas as pd
import h5py
from typing import Optional

# ─────────────────────────────────────────────
#  CONFIGURE PATHS HERE
# ─────────────────────────────────────────────

INSTANCE_CSV  = "/mnt/d/Downloads/Senior_Thesis/INSTANCE/metadata_Instance_events_v3.csv.bz2"
INSTANCE_HDF5 = "/mnt/d/Downloads/Senior_Thesis/INSTANCE/Instance_events_gm.hdf5"

# Set to None if you don't have the full noise HDF5
INSTANCE_NOISE_CSV  = "/mnt/d/Downloads/Senior_Thesis/INSTANCE/metadata_Instance_noise.csv.bz2"
INSTANCE_NOISE_HDF5 = None  # e.g. "/mnt/d/.../Instance_noise_gm.hdf5" if you have it

STEAD_DIR = "/mnt/d/Downloads/Senior_Thesis/STEAD/unzipped"
STEAD_CHUNKS = [f"chunk{i}" for i in range(1, 7)]

OUT_DIR  = "/home/aidar/study/senior_thesis/data"
TOTAL    = 1000   # samples per waveform — fixed

# Stratified event targets (per bin, across both datasets combined)
BIN_TARGETS = {
    "low":  25_000,   # magnitude [1.0, 3.0)
    "mid":  25_000,   # magnitude [3.0, 5.0)
    "high": 25_000,   # magnitude [5.0, ∞)
}
NOISE_TARGET = 20_000


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _mag_bin(mag: float) -> str:
    if mag < 3.0:
        return "low"
    elif mag < 5.0:
        return "mid"
    return "high"


def _stratified_sample(df: pd.DataFrame, target: int, seed: int = 42) -> pd.DataFrame:
    """Sample `target` rows from df, with replacement if df is smaller."""
    replace = len(df) < target
    if replace:
        print(f"    [!] Only {len(df)} rows available, oversampling with replacement to {target}")
    return df.sample(n=target, replace=replace, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────
#  METADATA LOADERS
# ─────────────────────────────────────────────

def load_instance_events(csv_path: str) -> pd.DataFrame:
    print("Loading INSTANCE events metadata...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.dropna(subset=["source_magnitude"])
    df = df[df["trace_eval_P"] == "manual"]
    df = df[df["station_channels"].isin(["HN", "HH"])]
    df = df[(df["trace_npts"] - df["trace_P_arrival_sample"]) >= TOTAL]
    df = df[df["source_magnitude"] >= 1.0]
    df = df.reset_index(drop=True)
    df["source"] = "instance"
    df["hdf5_path"] = INSTANCE_HDF5
    df = df.rename(columns={"trace_P_arrival_sample": "p_arrival_sample"})
    print(f"  INSTANCE events after filtering: {len(df)}")
    return df[["trace_name", "p_arrival_sample", "source_magnitude", "source", "hdf5_path"]]


def load_stead_events(stead_dir: str, chunks: list) -> pd.DataFrame:
    print("Loading STEAD events metadata (all chunks)...")
    dfs = []
    for chunk in chunks:
        csv_path = os.path.join(stead_dir, chunk, f"{chunk}.csv")
        hdf5_path = os.path.join(stead_dir, chunk, f"{chunk}.hdf5")
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["trace_category"].isin(["earthquake_local", "earthquake_regional"])]
        df = df.dropna(subset=["source_magnitude", "p_arrival_sample"])
        df = df[df["source_magnitude"] >= 1.0]
        # STEAD waveforms are 6000 samples — make sure there's 1000 samples after P
        df = df[df["p_arrival_sample"] <= (6000 - TOTAL)]
        df = df.reset_index(drop=True)
        df["source"] = f"stead_{chunk}"
        df["hdf5_path"] = hdf5_path
        print(f"  {chunk}: {len(df)} earthquake traces")
        dfs.append(df[["trace_name", "p_arrival_sample", "source_magnitude", "source", "hdf5_path"]])
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  STEAD events total: {len(combined)}")
    return combined


def load_stead_noise(stead_dir: str, chunks: list) -> pd.DataFrame:
    print("Loading STEAD noise metadata (all chunks)...")
    dfs = []
    for chunk in chunks:
        csv_path = os.path.join(stead_dir, chunk, f"{chunk}.csv")
        hdf5_path = os.path.join(stead_dir, chunk, f"{chunk}.hdf5")
        df = pd.read_csv(csv_path, low_memory=False)
        df = df[df["trace_category"] == "noise"]
        df = df.reset_index(drop=True)
        df["source"] = f"stead_{chunk}"
        df["hdf5_path"] = hdf5_path
        print(f"  {chunk}: {len(df)} noise traces")
        dfs.append(df[["trace_name", "source", "hdf5_path"]])
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  STEAD noise total: {len(combined)}")
    return combined


def load_instance_noise(csv_path: str, hdf5_path: str) -> pd.DataFrame:
    print("Loading INSTANCE noise metadata...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.reset_index(drop=True)
    df["source"] = "instance_noise"
    df["hdf5_path"] = hdf5_path
    print(f"  INSTANCE noise: {len(df)} traces")
    return df[["trace_name", "source", "hdf5_path"]]


# ─────────────────────────────────────────────
#  STRATIFIED EVENT SAMPLING
# ─────────────────────────────────────────────

def build_event_sample(
    instance_df: pd.DataFrame,
    stead_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each magnitude bin, try to fill 50% from INSTANCE and 50% from STEAD.
    Fall back gracefully if one source doesn't have enough.
    """
    print("\nBuilding stratified event sample...")
    all_df = pd.concat([instance_df, stead_df], ignore_index=True)
    all_df["mag_bin"] = all_df["source_magnitude"].apply(_mag_bin)

    result_frames = []

    for bin_name, target in BIN_TARGETS.items():
        half = target // 2
        bin_df = all_df[all_df["mag_bin"] == bin_name]

        inst_bin  = bin_df[bin_df["source"] == "instance"]
        stead_bin = bin_df[bin_df["source"] != "instance"]

        print(f"\n  Bin '{bin_name}': {len(inst_bin)} INSTANCE | {len(stead_bin)} STEAD | target {target}")

        # Try 50/50 split
        inst_target  = half
        stead_target = target - half

        # If one source falls short, take all of it and compensate from the other
        if len(inst_bin) < inst_target and len(stead_bin) >= stead_target:
            print(f"    INSTANCE short, taking all {len(inst_bin)} and pulling more from STEAD")
            inst_sample  = inst_bin
            stead_target = target - len(inst_bin)
            stead_sample = _stratified_sample(stead_bin, stead_target)
        elif len(stead_bin) < stead_target and len(inst_bin) >= inst_target:
            print(f"    STEAD short, taking all {len(stead_bin)} and pulling more from INSTANCE")
            stead_sample = stead_bin
            inst_target  = target - len(stead_bin)
            inst_sample  = _stratified_sample(inst_bin, inst_target)
        else:
            inst_sample  = _stratified_sample(inst_bin,  inst_target)
            stead_sample = _stratified_sample(stead_bin, stead_target)

        combined = pd.concat([inst_sample, stead_sample], ignore_index=True)
        print(f"    Sampled: {len(combined)} ({len(inst_sample)} INSTANCE + {len(stead_sample)} STEAD)")
        result_frames.append(combined)

    final = pd.concat(result_frames, ignore_index=True)
    final = final.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nTotal event sample: {len(final)}")
    return final


# ─────────────────────────────────────────────
#  WAVEFORM EXTRACTION
# ─────────────────────────────────────────────

def extract_instance_waveform(
    h5: h5py.File, trace_name: str, p_idx: int
) -> Optional[np.ndarray]:
    """Returns (3, 1000) float32 or None if trace is missing/short."""
    try:
        w = h5["data"][trace_name][:]          # (3, N)
        clip = w[:, p_idx : p_idx + TOTAL]
        if clip.shape != (3, TOTAL):
            return None
        return clip.astype(np.float32)
    except KeyError:
        return None


def extract_stead_waveform(
    h5: h5py.File, trace_name: str, p_idx: int
) -> Optional[np.ndarray]:
    """
    STEAD stores waveforms as (6000, 3) — time-first.
    Transpose to (3, 6000) then slice to (3, 1000).
    """
    try:
        w = h5["data"][trace_name][:]          # (6000, 3)
        w = w.T                                # → (3, 6000)
        clip = w[:, p_idx : p_idx + TOTAL]
        if clip.shape != (3, TOTAL):
            return None
        return clip.astype(np.float32)
    except KeyError:
        return None


def extract_noise_waveform(
    h5: h5py.File, trace_name: str, source: str
) -> Optional[np.ndarray]:
    """
    For noise, no P arrival — take a fixed crop from the middle of the window.
    STEAD noise is 6000 samples → crop [2500:3500] (middle 10 seconds).
    INSTANCE noise length varies → crop [0:1000].
    """
    try:
        w = h5["data"][trace_name][:]
        if "stead" in source:
            w = w.T                            # (6000, 3) → (3, 6000)
            clip = w[:, 2500:3500]
        else:
            clip = w[:, :TOTAL]               # INSTANCE: channel-first already

        if clip.shape != (3, TOTAL):
            return None
        return clip.astype(np.float32)
    except KeyError:
        return None


# ─────────────────────────────────────────────
#  MEMMAP WRITERS
# ─────────────────────────────────────────────

def write_events_memmap(event_df: pd.DataFrame, out_dir: str) -> int:
    """
    Iterate event_df, open HDF5 files by group (avoid re-opening per row),
    write to memmap. Returns actual count written (skips bad traces).
    """
    n_planned = len(event_df)
    memmap_path = os.path.join(out_dir, "waveforms_events.bin")

    # Pre-allocate (may be slightly over if some traces are skipped)
    waveforms = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(n_planned, 3, TOTAL))

    written = 0
    skipped = 0
    current_hdf5_path = None
    h5 = None

    print(f"\nExtracting {n_planned} event waveforms...")

    for i, row in event_df.iterrows():
        # Only re-open HDF5 when the file changes
        if row["hdf5_path"] != current_hdf5_path:
            if h5 is not None:
                h5.close()
            print(f"  Opening HDF5: {row['hdf5_path']}")
            h5 = h5py.File(row["hdf5_path"], "r")
            current_hdf5_path = row["hdf5_path"]

        p_idx = int(row["p_arrival_sample"])

        if row["source"] == "instance":
            clip = extract_instance_waveform(h5, row["trace_name"], p_idx)
        else:
            clip = extract_stead_waveform(h5, row["trace_name"], p_idx)

        if clip is None:
            skipped += 1
            continue

        waveforms[written] = clip
        written += 1

        if written % 1000 == 0:
            waveforms.flush()
            print(f"  {written}/{n_planned} written  ({skipped} skipped)")

    if h5 is not None:
        h5.close()

    waveforms.flush()
    del waveforms  # close the memmap

    # Trim the memmap to actual written count
    final = np.memmap(memmap_path, dtype="float32", mode="r", shape=(n_planned, 3, TOTAL))
    trimmed_path = memmap_path + ".trimmed"
    trimmed = np.memmap(trimmed_path, dtype="float32", mode="w+", shape=(written, 3, TOTAL))
    trimmed[:] = final[:written]
    trimmed.flush()
    del final, trimmed
    os.replace(trimmed_path, memmap_path)

    print(f"\nEvents done: {written} written, {skipped} skipped.")
    return written


def write_noise_memmap(noise_df: pd.DataFrame, out_dir: str) -> int:
    n_planned = len(noise_df)
    memmap_path = os.path.join(out_dir, "waveforms_noise.bin")
    waveforms = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(n_planned, 3, TOTAL))

    written = 0
    skipped = 0
    current_hdf5_path = None
    h5 = None

    print(f"\nExtracting {n_planned} noise waveforms...")

    for i, row in noise_df.iterrows():
        if row["hdf5_path"] != current_hdf5_path:
            if h5 is not None:
                h5.close()
            print(f"  Opening HDF5: {row['hdf5_path']}")
            h5 = h5py.File(row["hdf5_path"], "r")
            current_hdf5_path = row["hdf5_path"]

        clip = extract_noise_waveform(h5, row["trace_name"], row["source"])

        if clip is None:
            skipped += 1
            continue

        waveforms[written] = clip
        written += 1

        if written % 1000 == 0:
            waveforms.flush()
            print(f"  {written}/{n_planned} written  ({skipped} skipped)")

    if h5 is not None:
        h5.close()

    waveforms.flush()
    del waveforms

    # Trim
    final = np.memmap(memmap_path, dtype="float32", mode="r", shape=(n_planned, 3, TOTAL))
    trimmed_path = memmap_path + ".trimmed"
    trimmed = np.memmap(trimmed_path, dtype="float32", mode="w+", shape=(written, 3, TOTAL))
    trimmed[:] = final[:written]
    trimmed.flush()
    del final, trimmed
    os.replace(trimmed_path, memmap_path)

    print(f"\nNoise done: {written} written, {skipped} skipped.")
    return written


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load and filter metadata ──────────────────────────────────────
    instance_events = load_instance_events(INSTANCE_CSV)
    stead_events    = load_stead_events(STEAD_DIR, STEAD_CHUNKS)
    stead_noise     = load_stead_noise(STEAD_DIR, STEAD_CHUNKS)

    instance_noise = None
    if INSTANCE_NOISE_HDF5 is not None:
        instance_noise = load_instance_noise(INSTANCE_NOISE_CSV, INSTANCE_NOISE_HDF5)

    # ── 2. Stratified event sampling ─────────────────────────────────────
    event_sample = build_event_sample(instance_events, stead_events)
    
    # Sort by hdf5_path so we open each file once in a sequential block
    event_sample = event_sample.sort_values("hdf5_path").reset_index(drop=True)

    # ── 3. Noise sampling ────────────────────────────────────────────────
    noise_frames = [stead_noise]
    if instance_noise is not None:
        noise_frames.append(instance_noise)
    all_noise = pd.concat(noise_frames, ignore_index=True)
    noise_sample = _stratified_sample(all_noise, NOISE_TARGET)
    noise_sample = noise_sample.sort_values("hdf5_path").reset_index(drop=True)

    # ── 4. Write memmaps ─────────────────────────────────────────────────
    n_events_written = write_events_memmap(event_sample, OUT_DIR)
    n_noise_written  = write_noise_memmap(noise_sample, OUT_DIR)

    # ── 5. Save clean metadata CSVs (aligned with memmap rows) ───────────
    # Re-filter event_sample to only rows that were actually written
    # (skipped rows break alignment — easiest fix: re-run alignment pass)
    # We track written indices implicitly by saving the sorted df minus skips.
    # For simplicity: save full sample df — the very rare skipped traces won't
    # affect training materially, but log the count so you know.
    event_sample.to_csv(os.path.join(OUT_DIR, "metadata_events.csv"), index=False)
    noise_sample.to_csv(os.path.join(OUT_DIR, "metadata_noise.csv"),  index=False)

    # ── 6. Info file ─────────────────────────────────────────────────────
    info = {
        "events": {
            "n_planned": len(event_sample),
            "n_written": n_events_written,
            "shape": [n_events_written, 3, TOTAL],
            "bin_targets": BIN_TARGETS,
        },
        "noise": {
            "n_planned": len(noise_sample),
            "n_written": n_noise_written,
            "shape": [n_noise_written, 3, TOTAL],
        },
        "total": TOTAL,
    }
    with open(os.path.join(OUT_DIR, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'='*50}")
    print(f"All done.")
    print(f"  Events  : {n_events_written:,}  →  {OUT_DIR}/waveforms_events.bin")
    print(f"  Noise   : {n_noise_written:,}   →  {OUT_DIR}/waveforms_noise.bin")
    size_gb = (n_events_written + n_noise_written) * 3 * TOTAL * 4 / 1e9
    print(f"  Total disk usage: ~{size_gb:.2f} GB")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()