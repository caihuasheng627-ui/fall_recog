"""CSV columns, action labels, and event fields shared by the whole group."""

from __future__ import annotations

import re
from dataclasses import dataclass

COLUMNS = [
    "t_ms",
    "frame_id",
    "gear",
    "rng",
    "vel",
    "snr",
    "human",
    "lb",
    "ub",
    "hgt",
    "iq",
    "label",
]

REQUIRED_COLUMNS = [name for name in COLUMNS if name not in ("hgt", "iq", "label")]

IQ_BINS = 20
IQ_LEN = IQ_BINS * 2

LABELS = ("empty", "stand", "walk", "sit", "bend", "lie", "fall")

LABEL_ZH = {
    "empty": "空房间",
    "stand": "站立",
    "walk": "走动",
    "sit": "坐下",
    "bend": "弯腰",
    "lie": "主动躺下",
    "fall": "跌倒",
}

STATES = ("normal", "suspect", "alarm")

EVENT_KEYS = (
    "device_id",
    "t_ms",
    "frame_id",
    "state",
    "action",
    "reason",
    "rng",
    "vel",
    "human",
)

FILENAME_RE = re.compile(r"^(P\d+)_([a-z]+)_(\d+)_(train|test)$")

REASONS = ("idle", "fast_then_still", "no_human", "recovered", "cleared")


@dataclass
class Frame:
    t_ms: int
    frame_id: int
    gear: int
    rng: int
    vel: int
    snr: float
    human: int
    lb: int
    ub: int
    label: str
    hgt: int | None = None
    iq: tuple[int, ...] | None = None


def parse_filename(stem: str) -> dict[str, str] | None:
    matched = FILENAME_RE.match(stem)
    if matched is None:
        return None
    person, action, seq, split = matched.groups()
    if action not in LABELS:
        raise ValueError(f"文件名动作无效: {stem}")
    return {"person": person, "action": action, "seq": seq, "split": split}


def gear_from_rng_cm(rng: int, human: int) -> tuple[int, int, int]:
    """Map a centimeter range to gear, lb, ub for synthetic logs only.

    The sample treats 0–100 as the 1 m bin and 100–200 as the 2 m bin.
    Real board logs must keep the raw gear, lb, and ub. Do not recompute them.
    """
    if human == 0:
        return 0, 0, 0
    if rng < 100:
        return 0, 0, 100
    if rng < 200:
        return 1, 100, 200
    return 2, 200, 600


def synthetic_iq(rng: int, vel: int, human: int, snr: float) -> tuple[int, ...]:
    """Place a peak in the range bin that matches rng. Board logs must store real IQ."""
    values = [0] * IQ_LEN
    if human == 0:
        return tuple(values)
    bin_index = min(IQ_BINS - 1, max(0, int(rng) // 30))
    mag = max(1, int(abs(snr) * 40))
    values[2 * bin_index] = mag
    values[2 * bin_index + 1] = int(vel)
    return tuple(values)
