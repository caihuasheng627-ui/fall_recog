"""Sliding-window features: I/Q magnitudes first, then gear/rng/vel/snr."""

from __future__ import annotations

import math
import statistics

from fall_recog.schema import IQ_BINS, IQ_LEN, Frame

IQ_FEATURE_NAMES = tuple(f"iq_mag_{index:02d}" for index in range(IQ_BINS))
AUX_FEATURE_NAMES = (
    "n_valid",
    "duration_s",
    "rng_delta",
    "rng_min",
    "rng_max",
    "rng_mean",
    "rng_std",
    "rng_slope",
    "vel_mean",
    "vel_abs_mean",
    "vel_abs_max",
    "vel_std",
    "vel_sign",
    "snr_mean",
    "snr_std",
    "human_frac",
    "gear_changes",
    "fast_vel_frac",
    "iq_peak_bin",
    "iq_peak_mag",
    "iq_energy",
)
FEATURE_NAMES = IQ_FEATURE_NAMES + AUX_FEATURE_NAMES
MIN_VALID = 2


def _iq_mags(frame: Frame) -> list[float] | None:
    if frame.iq is None or len(frame.iq) != IQ_LEN:
        return None
    mags = []
    for index in range(IQ_BINS):
        real = float(frame.iq[2 * index])
        imag = float(frame.iq[2 * index + 1])
        mags.append(math.hypot(real, imag))
    return mags


def extract_features(
    frames: list[Frame],
    invalid_value: int,
    vel_fall: float,
) -> dict[str, float] | None:
    valid = [frame for frame in frames if frame.rng != invalid_value and frame.vel != invalid_value]
    if len(valid) < MIN_VALID:
        return None
    rngs = [float(frame.rng) for frame in valid]
    vels = [float(frame.vel) for frame in valid]
    snrs = [float(frame.snr) for frame in valid]
    t0 = valid[0].t_ms
    t1 = valid[-1].t_ms
    duration_s = max((t1 - t0) / 1000.0, 0.001)
    abs_vels = [abs(value) for value in vels]
    vel_abs_sum = sum(abs_vels) or 1.0
    gears = [frame.gear for frame in valid]
    gear_changes = float(sum(1 for prev, cur in zip(gears, gears[1:]) if prev != cur))

    mag_sum = [0.0] * IQ_BINS
    mag_count = 0
    for frame in valid:
        mags = _iq_mags(frame)
        if mags is None:
            continue
        mag_count += 1
        for index, value in enumerate(mags):
            mag_sum[index] += value
    if mag_count:
        iq_mean = [value / mag_count for value in mag_sum]
    else:
        iq_mean = [0.0] * IQ_BINS
    peak_bin = max(range(IQ_BINS), key=lambda index: iq_mean[index])
    peak_mag = iq_mean[peak_bin]
    energy = sum(iq_mean)

    features = {name: iq_mean[index] for index, name in enumerate(IQ_FEATURE_NAMES)}
    features.update(
        {
            "n_valid": float(len(valid)),
            "duration_s": duration_s,
            "rng_delta": rngs[-1] - rngs[0],
            "rng_min": min(rngs),
            "rng_max": max(rngs),
            "rng_mean": statistics.fmean(rngs),
            "rng_std": statistics.pstdev(rngs),
            "rng_slope": (rngs[-1] - rngs[0]) / duration_s,
            "vel_mean": statistics.fmean(vels),
            "vel_abs_mean": statistics.fmean(abs_vels),
            "vel_abs_max": max(abs_vels),
            "vel_std": statistics.pstdev(vels),
            "vel_sign": sum(vels) / vel_abs_sum,
            "snr_mean": statistics.fmean(snrs),
            "snr_std": statistics.pstdev(snrs),
            "human_frac": sum(1 for frame in valid if frame.human == 1) / len(valid),
            "gear_changes": gear_changes,
            "fast_vel_frac": sum(1 for value in abs_vels if value >= vel_fall) / len(valid),
            "iq_peak_bin": float(peak_bin),
            "iq_peak_mag": peak_mag,
            "iq_energy": energy,
        }
    )
    return features


def feature_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]


def window_is_empty(frames: list[Frame]) -> bool:
    if not frames:
        return True
    return all(frame.human == 0 for frame in frames)
