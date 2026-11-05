"""Read and write the radar CSV agreed with the capture side."""

from __future__ import annotations

import csv
from pathlib import Path

from fall_recog.schema import COLUMNS, IQ_LEN, REQUIRED_COLUMNS, LABELS, Frame, parse_filename

GAP_WARN_MS = 250


class CsvError(ValueError):
    pass


def read_header_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            for part in stripped[1:].split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    meta[key] = value
    return meta


def read_sidecar(path: Path) -> dict[str, str]:
    sidecar = path.with_suffix(".txt")
    meta: dict[str, str] = {}
    if not sidecar.exists():
        return meta
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def _data_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(line)
    if not lines:
        raise CsvError(f"{path} 没有数据行")
    return lines


def read_frames(path: Path) -> tuple[list[Frame], list[str]]:
    path = Path(path)
    warnings: list[str] = []
    reader = csv.DictReader(_data_lines(path))
    if reader.fieldnames is None:
        raise CsvError(f"{path} 缺少表头")
    fieldnames = [name.strip() for name in reader.fieldnames]
    reader.fieldnames = fieldnames
    missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
    if missing:
        raise CsvError(f"{path} 缺少列: {', '.join(missing)}")

    named = parse_filename(path.stem)
    frames: list[Frame] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            label = (row.get("label") or "").strip()
            if not label:
                if named is None:
                    raise CsvError("缺少 label，且文件名不含动作")
                label = named["action"]
            if label not in LABELS:
                raise CsvError(f"非法 label: {label}")
            human = int(row["human"])
            if human not in (0, 1):
                raise CsvError(f"human 只能是 0 或 1，实际 {human}")
            hgt = _optional_hgt(row.get("hgt"))
            iq = _optional_iq(row.get("iq"))
            frame = Frame(
                t_ms=int(float(row["t_ms"])),
                frame_id=int(row["frame_id"]),
                gear=int(row["gear"]),
                rng=int(float(row["rng"])),
                vel=int(float(row["vel"])),
                snr=float(row["snr"]),
                human=human,
                lb=int(float(row["lb"])),
                ub=int(float(row["ub"])),
                label=label,
                hgt=hgt,
                iq=iq,
            )
        except CsvError:
            raise
        except (TypeError, ValueError) as exc:
            raise CsvError(f"{path} 第 {row_number} 行无法解析: {exc}") from exc
        frames.append(frame)

    if not frames:
        raise CsvError(f"{path} 没有数据行")

    labels = {frame.label for frame in frames}
    if len(labels) != 1:
        raise CsvError(f"{path} 含有多个 label: {sorted(labels)}")
    if named is not None and frames[0].label != named["action"]:
        raise CsvError(
            f"{path} 的 label={frames[0].label} 与文件名动作 {named['action']} 不一致"
        )
    if frames[0].t_ms != 0:
        warnings.append(f"t_ms 不是从 0 开始，而是 {frames[0].t_ms}")
    if frames[0].frame_id != 0:
        warnings.append(f"frame_id 不是从 0 开始，而是 {frames[0].frame_id}")

    gap_warned = False
    for prev, cur in zip(frames, frames[1:]):
        if cur.t_ms <= prev.t_ms:
            raise CsvError(f"{path} 的 t_ms 没有严格递增: {prev.t_ms} -> {cur.t_ms}")
        if cur.frame_id <= prev.frame_id:
            raise CsvError(
                f"{path} 的 frame_id 没有严格递增: {prev.frame_id} -> {cur.frame_id}"
            )
        gap = cur.t_ms - prev.t_ms
        if gap > GAP_WARN_MS and not gap_warned:
            warnings.append(f"帧间隔 {gap} ms，超过 {GAP_WARN_MS} ms（{prev.t_ms} -> {cur.t_ms}）")
            gap_warned = True
    return frames, warnings


def _optional_hgt(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text == "-999":
        return None
    return int(float(text))


def _optional_iq(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text == "-999":
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != IQ_LEN:
        raise CsvError(f"iq 必须是 {IQ_LEN} 个整数（I/Q 交替），实际 {len(parts)} 个")
    return tuple(int(float(part)) for part in parts)


def write_frames(path: Path, frames: list[Frame], comments: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for comment in comments:
            handle.write(f"# {comment}\n")
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "t_ms": frame.t_ms,
                    "frame_id": frame.frame_id,
                    "gear": frame.gear,
                    "rng": frame.rng,
                    "vel": frame.vel,
                    "snr": f"{frame.snr:.1f}",
                    "human": frame.human,
                    "lb": frame.lb,
                    "ub": frame.ub,
                    "hgt": -999 if frame.hgt is None else frame.hgt,
                    "iq": -999 if frame.iq is None else " ".join(str(value) for value in frame.iq),
                    "label": frame.label,
                }
            )


def write_sidecar(path: Path, meta: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in meta.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_recordings(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    files = []
    for csv_path in sorted(path.rglob("*.csv")):
        try:
            named = parse_filename(csv_path.stem)
        except ValueError as exc:
            raise CsvError(str(exc)) from exc
        if named is not None:
            files.append(csv_path)
    return files
