"""Plot 20 range-bin I/Q magnitudes over time. Bin index is range, not height."""

from __future__ import annotations

from pathlib import Path

from fall_recog.io_csv import read_frames
from fall_recog.schema import IQ_BINS, IQ_LEN, Frame


def frame_mags(frame: Frame) -> list[float] | None:
    if frame.iq is None or len(frame.iq) != IQ_LEN:
        return None
    mags = []
    for index in range(IQ_BINS):
        real = float(frame.iq[2 * index])
        imag = float(frame.iq[2 * index + 1])
        mags.append((real * real + imag * imag) ** 0.5)
    return mags


def matrix(frames: list[Frame]) -> tuple[list[int], list[list[float]]]:
    times: list[int] = []
    rows: list[list[float]] = []
    for frame in frames:
        mags = frame_mags(frame)
        if mags is None:
            continue
        times.append(frame.t_ms)
        rows.append(mags)
    return times, rows


def plot_recording(csv_path: Path, out_path: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("画图需要 matplotlib：pip install matplotlib") from exc
    frames, _ = read_frames(csv_path)
    times, rows = matrix(frames)
    if not rows:
        raise ValueError(f"{csv_path} 没有 iq 列，无法画距离门。板端要落原始 I/Q。")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = list(zip(*rows))
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        extent=(times[0], times[-1], 0, IQ_BINS),
        interpolation="nearest",
    )
    ax.set_xlabel("t_ms")
    ax.set_ylabel("range bin (0=near ... 19=far)")
    ax.set_title(f"{csv_path.name}  I/Q magnitude  (bin is range, not height)")
    fig.colorbar(image, ax=ax, label="|I+jQ|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
