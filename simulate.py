"""Synthetic recordings so the recognizer can be tested before the board logs exist."""

from __future__ import annotations

import math
import random
import zlib
from pathlib import Path

from fall_recog.io_csv import write_frames, write_sidecar
from fall_recog.schema import LABELS, Frame, gear_from_rng_cm, synthetic_iq

DT_MS = 50
ROOT = Path(__file__).resolve().parent
H_STAND = 150.0
H_SIT = 85.0
H_LIE = 25.0

PEOPLE = (
    {
        "person": "P01",
        "split": "test",
        "height_cm": "168",
        "distance_m": "1.5",
        "radar_height_m": "1.8",
        "facing": "front",
    },
    {
        "person": "P02",
        "split": "train",
        "height_cm": "175",
        "distance_m": "1.2",
        "radar_height_m": "1.8",
        "facing": "front",
    },
    {
        "person": "P03",
        "split": "train",
        "height_cm": "160",
        "distance_m": "2.0",
        "radar_height_m": "1.8",
        "facing": "front",
    },
)

SEQ_SCALE = {1: 0.92, 2: 1.0, 3: 1.08}


def _seed(*parts: str) -> int:
    text = "|".join(parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _still(
    duration_ms: int,
    rng: float,
    human: int = 1,
    hgt: float | None = H_STAND,
) -> list[tuple[int, int, float, float, float | None]]:
    samples = []
    t_ms = 0
    while t_ms < duration_ms:
        samples.append((t_ms, human, rng, 0.0, hgt))
        t_ms += DT_MS
    return samples


def _bump(
    duration_ms: int,
    rng0: float,
    peak: float,
    hgt0: float,
    hgt1: float,
) -> tuple[list[tuple[int, int, float, float, float | None]], float]:
    samples = []
    t_ms = 0
    rng = rng0
    while t_ms < duration_ms:
        vel = peak * math.sin(math.pi * t_ms / duration_ms)
        progress = (1.0 - math.cos(math.pi * t_ms / duration_ms)) / 2.0
        hgt = hgt0 + (hgt1 - hgt0) * progress
        samples.append((t_ms, 1, rng, vel, hgt))
        rng = rng + vel * DT_MS / 1000.0
        t_ms += DT_MS
    return samples, rng


def _rush(
    duration_ms: int,
    rng0: float,
    vel: float,
    hgt: float,
) -> tuple[list[tuple[int, int, float, float, float | None]], float]:
    samples = []
    t_ms = 0
    rng = rng0
    while t_ms < duration_ms:
        samples.append((t_ms, 1, rng, vel, hgt))
        rng = rng + vel * DT_MS / 1000.0
        t_ms += DT_MS
    return samples, rng


def _walk(
    duration_ms: int,
    center: float,
    amp: float = 30.0,
    hgt: float = H_STAND,
) -> list[tuple[int, int, float, float, float | None]]:
    period_ms = 4000.0
    period_s = period_ms / 1000.0
    samples = []
    t_ms = 0
    while t_ms < duration_ms:
        angle = 2.0 * math.pi * t_ms / period_ms
        rng = center + amp * math.sin(angle)
        vel = amp * (2.0 * math.pi / period_s) * math.cos(angle)
        samples.append((t_ms, 1, rng, vel, hgt))
        t_ms += DT_MS
    return samples


def _bend(rng0: float, peak: float) -> tuple[list[tuple[int, int, float, float, float | None]], float]:
    parts: list[list[tuple[int, int, float, float, float | None]]] = []
    rng = rng0
    for _ in range(3):
        down, rng = _bump(800, rng, peak, H_STAND, 90.0)
        hold = _still(400, rng, hgt=90.0)
        up, rng = _bump(800, rng, -peak, 90.0, H_STAND)
        pause = _still(800, rng, hgt=H_STAND)
        parts.extend((down, hold, up, pause))
    return _chain(parts), rng


def _chain(
    parts: list[list[tuple[int, int, float, float, float | None]]],
) -> list[tuple[int, int, float, float, float | None]]:
    merged = []
    offset = 0
    for part in parts:
        if not part:
            continue
        for t_ms, human, rng, vel, hgt in part:
            merged.append((t_ms + offset, human, rng, vel, hgt))
        offset = merged[-1][0] + DT_MS
    return merged


def build_clean(action: str, rng0: float, scale: float, fall_peak: float, sit_peak: float) -> tuple[list, int | None]:
    if action == "empty":
        return _still(20000, 0.0, human=0, hgt=None), None
    if action == "stand":
        return _still(15000, rng0, human=1, hgt=H_STAND), None

    prep = _still(3000, rng0, human=1, hgt=H_STAND)
    onset = 3000
    if action == "walk":
        motion = _walk(20000, rng0, amp=30.0 * scale, hgt=H_STAND)
        return _chain([prep, motion]), onset
    if action == "sit":
        motion, rng1 = _bump(1200, rng0, sit_peak * scale, H_STAND, H_SIT)
        return _chain([prep, motion, _still(10000, rng1, hgt=H_SIT)]), onset
    if action == "bend":
        motion, rng1 = _bend(rng0, -50.0 * scale)
        return _chain([prep, motion, _still(2000, rng1, hgt=H_STAND)]), onset
    if action == "lie":
        motion, rng1 = _bump(3200, rng0, -36.0 * scale, H_STAND, H_LIE)
        return _chain([prep, motion, _still(12000, rng1, hgt=H_LIE)]), onset
    if action == "fall":
        motion, rng1 = _bump(700, rng0, fall_peak * scale, H_STAND, H_LIE)
        return _chain([prep, motion, _still(15000, rng1, hgt=H_LIE)]), onset
    raise ValueError(f"未知动作 {action}")


def materialize(
    samples: list[tuple[int, int, float, float, float | None]],
    rnd: random.Random,
    label: str,
) -> list[Frame]:
    frames: list[Frame] = []
    for index, (t_ms, human, rng, vel, hgt) in enumerate(samples):
        rng_n = rng + (0.0 if human == 0 and label == "empty" else rnd.gauss(0.0, 1.5))
        vel_n = vel + rnd.gauss(0.0, 4.0)
        human_n = human
        if human_n == 1 and abs(vel) < 30 and rnd.random() < 0.004:
            human_n = 0
        rng_i = int(round(rng_n))
        vel_i = int(round(vel_n))
        gear, lb, ub = gear_from_rng_cm(rng_i, human_n)
        if human_n == 1 and hgt is not None:
            hgt_i: int | None = int(round(hgt + rnd.gauss(0.0, 1.0)))
        else:
            hgt_i = None
        if human_n == 1:
            snr = min(40.0, max(8.0, 20.0 + rnd.gauss(0.0, 1.2)))
        else:
            snr = min(12.0, max(0.0, 5.0 + rnd.gauss(0.0, 0.6)))
        frames.append(
            Frame(
                t_ms=int(t_ms),
                frame_id=index,
                gear=gear,
                rng=rng_i,
                vel=vel_i,
                snr=round(snr, 1),
                human=human_n,
                lb=lb,
                ub=ub,
                label=label,
                hgt=hgt_i,
                iq=synthetic_iq(rng_i, vel_i, human_n, snr),
            )
        )
    return frames


def _write_recording(
    folder: Path,
    person: dict[str, str],
    action: str,
    seq: int,
    frames: list[Frame],
    onset: int | None,
    extra: dict[str, str] | None = None,
) -> Path:
    split = person["split"]
    stem = f"{person['person']}_{action}_{seq:02d}_{split}"
    csv_path = folder / split / f"{stem}.csv"
    if extra and extra.get("folder"):
        csv_path = folder / extra["folder"] / f"{stem}.csv"
    comments = [
        "source=synthetic rng_unit=cm vel_unit=cm/s hgt_unit=cm frame_period_ms=50",
        "hgt is synthetic body height; -999 means unknown. Board logs omit hgt until a real source exists",
        "iq is 40 ints I/Q interleaved (20 range bins). Synthetic places a peak at rng; board must dump raw IQ",
        "gear/lb/ub are a centimeter hypothesis for synthetic logs; board logs must store raw values",
    ]
    write_frames(csv_path, frames, comments)
    meta = {
        "person": person["person"],
        "action": action,
        "seq": f"{seq:02d}",
        "split": split,
        "height_cm": person["height_cm"],
        "distance_m": person["distance_m"],
        "radar_height_m": person["radar_height_m"],
        "facing": person["facing"],
        "mat": "yes",
        "source": "synthetic",
        "motion_onset_ms": "" if onset is None else str(onset),
        "note": "",
    }
    if extra:
        for key, value in extra.items():
            if key != "folder":
                meta[key] = value
    write_sidecar(csv_path.with_suffix(".txt"), meta)
    return csv_path


def generate(out_dir: Path | None = None, seed: int = 20261029) -> list[Path]:
    out_dir = Path(out_dir) if out_dir is not None else ROOT / "data" / "synthetic"
    written: list[Path] = []
    manifest_rows = ["person,action,seq,split,file,frames,duration_ms,source,note"]

    for person in PEOPLE:
        rng0 = float(person["distance_m"]) * 100.0
        for action in LABELS:
            for seq in (1, 2, 3):
                scale = SEQ_SCALE[seq]
                samples, onset = build_clean(action, rng0, scale, fall_peak=-150.0, sit_peak=-45.0)
                rnd = random.Random(_seed(str(seed), person["person"], action, str(seq)))
                frames = materialize(samples, rnd, action)
                path = _write_recording(out_dir, person, action, seq, frames, onset)
                written.append(path)
                manifest_rows.append(
                    f"{person['person']},{action},{seq:02d},{person['split']},{path.name},{len(frames)},{frames[-1].t_ms},synthetic,"
                )

    boundary_person = {
        "person": "P09",
        "split": "test",
        "height_cm": "170",
        "distance_m": "1.5",
        "radar_height_m": "1.8",
        "facing": "front",
    }
    rng0 = 150.0
    slow_motion, slow_rng = _bump(2800, rng0, -42.0, H_STAND, H_LIE)
    slow_samples = _chain([_still(3000, rng0, hgt=H_STAND), slow_motion, _still(15000, slow_rng, hgt=H_LIE)])
    rush, rush_rng = _rush(400, rng0, -100.0, H_STAND)
    stop_samples = _chain([_still(3000, rng0, hgt=H_STAND), rush, _still(10000, rush_rng, hgt=H_STAND)])
    cases = (
        (
            "sit",
            1,
            build_clean("sit", rng0, 1.0, fall_peak=-150.0, sit_peak=-150.0),
            "no_alarm",
            "快速坐下，速度和位移都达到跌倒门限，并坐稳超过 8 秒。高度停在坐姿，不应当报警",
        ),
        (
            "fall",
            1,
            (slow_samples, 3000),
            "no_alarm",
            "缓慢倒下，位移已经够大，高度也到了地面，但速度低于跌倒门限。和主动躺下是同一类轨迹，不能放宽门限",
        ),
        (
            "walk",
            1,
            (stop_samples, 3000),
            "no_alarm",
            "快走后急停，速度约 100、位移约 40 厘米，再站立超过 8 秒。高度没有下降，不应当报警",
        ),
    )
    for action, seq, built, expect, note in cases:
        samples, onset = built
        rnd = random.Random(_seed(str(seed), "boundary", action, str(seq)))
        frames = materialize(samples, rnd, action)
        path = _write_recording(
            out_dir,
            boundary_person,
            action,
            seq,
            frames,
            onset,
            extra={"folder": "boundary", "expect": expect, "note": note, "source": "synthetic-boundary"},
        )
        written.append(path)
        manifest_rows.append(
            f"P09,{action},{seq:02d},test,{path.name},{len(frames)},{frames[-1].t_ms},synthetic-boundary,{note}"
        )

    manifest = out_dir / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    return written
