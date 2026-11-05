"""Train the seven-class action model from labeled CSV recordings."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from fall_recog.features import FEATURE_NAMES, extract_features, feature_vector
from fall_recog.io_csv import iter_recordings, read_frames, read_sidecar
from fall_recog.schema import Frame, LABELS, parse_filename

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "models" / "action_rf.joblib"
WINDOW_MS = 2000
STEP_MS = 100
FALL_ACTION_MS = 2000
BEND_ACTION_MS = 8000


def window_label(file_label: str, window_end_ms: int, onset: int | None) -> str:
    if file_label in ("empty", "stand"):
        return file_label
    if onset is None:
        return file_label
    if window_end_ms < onset:
        return "stand" if file_label != "empty" else "empty"
    if file_label == "walk":
        return "walk"
    if file_label == "sit":
        return "sit"
    if file_label == "lie":
        return "lie"
    if file_label == "fall":
        if window_end_ms <= onset + FALL_ACTION_MS:
            return "fall"
        return "lie"
    if file_label == "bend":
        if window_end_ms <= onset + BEND_ACTION_MS:
            return "bend"
        return "stand"
    return file_label


def iter_windows(frames: list[Frame], window_ms: int, step_ms: int) -> list[list[Frame]]:
    if not frames:
        return []
    t0 = frames[0].t_ms
    t_end = frames[-1].t_ms
    windows: list[list[Frame]] = []
    start = t0
    while start <= t_end:
        stop = start + window_ms
        chunk = [frame for frame in frames if start <= frame.t_ms <= stop]
        if chunk:
            windows.append(chunk)
        start += step_ms
        if start > t_end:
            break
    return windows


def collect_samples(
    input_dir: Path,
    invalid_value: int,
    vel_fall: float,
    window_ms: int = WINDOW_MS,
    step_ms: int = STEP_MS,
) -> tuple[list[list[float]], list[str]]:
    xs: list[list[float]] = []
    ys: list[str] = []
    for path in iter_recordings(input_dir):
        if "boundary" in path.parts:
            continue
        named = parse_filename(path.stem)
        if named is not None and named["person"] == "P01":
            continue
        frames, _ = read_frames(path)
        sidecar = read_sidecar(path)
        onset_text = sidecar.get("motion_onset_ms", "")
        onset = int(onset_text) if onset_text.isdigit() else None
        file_label = frames[0].label
        for chunk in iter_windows(frames, window_ms, step_ms):
            features = extract_features(chunk, invalid_value, vel_fall)
            if features is None:
                continue
            if all(frame.human == 0 for frame in chunk):
                label = "empty"
            else:
                label = window_label(file_label, chunk[-1].t_ms, onset)
            if label not in LABELS:
                continue
            xs.append(feature_vector(features))
            ys.append(label)
    return xs, ys


def train_estimator(xs: list[list[float]], ys: list[str]):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError("需要安装 scikit-learn：pip install scikit-learn joblib") from exc
    if not xs:
        raise ValueError("没有可用的训练窗")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(xs, ys)
    return clf


def save_model(estimator, path: Path) -> None:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("需要安装 joblib：pip install scikit-learn joblib") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": estimator,
            "feature_names": FEATURE_NAMES,
            "classes": list(estimator.classes_),
        },
        path,
    )


def train_dir(
    input_dir: Path,
    out_path: Path,
    invalid_value: int = -999,
    vel_fall: float = 90.0,
) -> dict:
    xs, ys = collect_samples(input_dir, invalid_value, vel_fall)
    counts = dict(Counter(ys))
    estimator = train_estimator(xs, ys)
    save_model(estimator, out_path)
    acc = float(estimator.score(xs, ys))
    return {"samples": len(ys), "counts": counts, "train_acc": acc, "path": out_path}
