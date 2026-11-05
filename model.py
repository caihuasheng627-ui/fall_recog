"""Load and run the action classifier."""

from __future__ import annotations

from pathlib import Path

from fall_recog.features import FEATURE_NAMES, extract_features, feature_vector, window_is_empty
from fall_recog.schema import Frame, LABELS

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "action_rf.joblib"


class ModelError(ValueError):
    pass


class ActionClassifier:
    def __init__(self, estimator, classes: tuple[str, ...]):
        self.estimator = estimator
        self.classes = classes
        index = {name: i for i, name in enumerate(estimator.classes_)}
        self._fall_index = index.get("fall")

    @classmethod
    def load(cls, path: Path | str) -> "ActionClassifier":
        try:
            import joblib
        except ImportError as exc:
            raise ModelError("需要安装 scikit-learn 和 joblib：pip install scikit-learn joblib") from exc
        payload = joblib.load(Path(path))
        names = tuple(payload.get("feature_names", FEATURE_NAMES))
        if names != FEATURE_NAMES:
            raise ModelError("模型特征与当前程序不一致，请重新训练")
        return cls(payload["estimator"], tuple(payload.get("classes", LABELS)))

    def predict(
        self,
        frames: list[Frame],
        invalid_value: int,
        vel_fall: float,
    ) -> tuple[str, float] | None:
        if window_is_empty(frames):
            return "empty", 1.0
        features = extract_features(frames, invalid_value, vel_fall)
        if features is None:
            return None
        vector = [feature_vector(features)]
        label = str(self.estimator.predict(vector)[0])
        proba = self.estimator.predict_proba(vector)[0]
        if self._fall_index is None:
            return label, 0.0
        return label, float(proba[self._fall_index])


def resolve_model_path(path: str | Path, params_dir: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    root = ROOT
    relative = root / candidate
    if relative.exists():
        return relative
    if params_dir is not None:
        other = Path(params_dir) / candidate
        if other.exists():
            return other
    return relative
