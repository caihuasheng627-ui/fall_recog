"""Fall rule: a fast, same-direction range change, then stillness.

Thresholds live in params.json and use CSV units. Frames with rng or vel
equal to invalid_value are dropped. The detector never reads the label column.

When a frame carries hgt, stillness only counts as a fall if the height has
dropped to a lying level. A range rebound toward the pre-impact distance
cancels the impact even when height is missing. Continued motion in the
impact direction is still the fall, not a cancel. Alarm latches until a
clear command; human dropouts do not reset an in-progress impact. vel_fall
and impact_rng are not lowered to catch slow descents: that also flags an
intentional lie-down.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from fall_recog.schema import EVENT_KEYS, Frame

REQUIRED_PARAMS = (
    "device_id",
    "vel_fall",
    "still_vel",
    "recover_vel",
    "min_impact_frames",
    "impact_span_ms",
    "impact_rng",
    "return_rng",
    "rebound_rng",
    "motion_cancel_ms",
    "recent_ms",
    "still_suspect_ms",
    "still_alarm_ms",
    "absent_reset_ms",
    "invalid_value",
    "height_drop",
    "height_lie_max",
)


class CommandError(ValueError):
    """The uplink line is not a clear command for this device."""


def load_params(path: Path) -> dict:
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_PARAMS if key not in params]
    if missing:
        raise ValueError(f"params.json 缺少字段: {', '.join(missing)}")
    if params["still_suspect_ms"] >= params["still_alarm_ms"]:
        raise ValueError("still_suspect_ms 必须小于 still_alarm_ms")
    if params["still_vel"] >= params["recover_vel"]:
        raise ValueError("still_vel 必须小于 recover_vel")
    if params["recover_vel"] >= params["vel_fall"]:
        raise ValueError("recover_vel 必须小于 vel_fall")
    if params["return_rng"] >= params["impact_rng"]:
        raise ValueError("return_rng 必须小于 impact_rng")
    if params["rebound_rng"] <= params["return_rng"]:
        raise ValueError("rebound_rng 必须大于 return_rng")
    if params["motion_cancel_ms"] <= 0:
        raise ValueError("motion_cancel_ms 必须大于 0")
    if params["impact_rng"] <= 0:
        raise ValueError("impact_rng 必须大于 0")
    if params["height_drop"] <= 0:
        raise ValueError("height_drop 必须大于 0")
    if params["height_lie_max"] <= 0:
        raise ValueError("height_lie_max 必须大于 0")
    return params


def make_event(params: dict, frame: Frame, state: str, action: str, reason: str) -> dict:
    event = {
        "device_id": params["device_id"],
        "t_ms": int(frame.t_ms),
        "frame_id": int(frame.frame_id),
        "state": state,
        "action": action,
        "reason": reason,
        "rng": int(frame.rng),
        "vel": int(frame.vel),
        "human": int(frame.human),
    }
    return {key: event[key] for key in EVENT_KEYS}


class FallDetector:
    """Fall-like window, then stillness. Optional 7-class classifier."""

    def __init__(self, params: dict, classifier=None):
        self.params = params
        self.classifier = classifier
        if self.classifier is None and params.get("use_classifier"):
            from fall_recog.model import ActionClassifier, resolve_model_path

            model_path = resolve_model_path(params.get("model_path", "models/action_rf.joblib"))
            if not model_path.exists():
                raise ValueError(f"找不到分类器 {model_path}，先运行 python -m fall_recog train")
            self.classifier = ActionClassifier.load(model_path)
        self.reset()

    def reset(self) -> None:
        self.phase = "normal"
        self.last_emit: tuple[str, str] | None = None
        self.started = False
        self.window: deque[Frame] = deque()
        self.last_pred: str | None = None
        self._reset_track()

    def _reset_track(self) -> None:
        self.impact_on = False
        self.impact_peak = 0
        self.impact_sign = 0
        self.rng_before = 0
        self.rng_far = 0
        self.hgt_before: int | None = None
        self.posture_bad = 0
        self.still_acc = 0
        self.motion_acc = 0
        self.last_t: int | None = None
        self.paused = False
        self.recover_high = 0
        self.absent_since: int | None = None
        self.cancel_windows = 0
        self.recent: deque[tuple[int, int, int, int | None]] = deque()

    def clear(self, frame: Frame) -> dict:
        self._reset_track()
        return self._emit(frame, "normal", "none", "cleared", force=True)

    def push(self, frame: Frame) -> list[dict]:
        events: list[dict] = []
        if not self.started:
            self.started = True
            events.append(self._emit(frame, "normal", "none", "idle", force=True))

        if frame.human == 0:
            if self.phase == "alarm":
                return events
            if self.impact_on:
                self.paused = True
                return events
            events.extend(self._on_absent(frame))
            return events

        self.absent_since = None
        if self.phase == "alarm":
            return events

        if self._invalid(frame):
            if self.impact_on:
                self.paused = True
            return events

        self.window.append(frame)
        window_ms = int(self.params.get("window_ms", 2000))
        while self.window and frame.t_ms - self.window[0].t_ms > window_ms:
            self.window.popleft()

        self.recent.append((frame.t_ms, frame.vel, frame.rng, frame.hgt))
        while self.recent and frame.t_ms - self.recent[0][0] > self.params["recent_ms"]:
            self.recent.popleft()

        if not self.impact_on:
            if self._confirm_impact(frame):
                self.last_t = frame.t_ms
                self.paused = False
                self.still_acc = 0
                self.motion_acc = 0
                self.recover_high = 0
                self.posture_bad = 0
            return events

        events.extend(self._settle(frame))
        return events

    def run(self, frames: list[Frame]) -> list[dict]:
        self.reset()
        events: list[dict] = []
        for frame in frames:
            events.extend(self.push(frame))
        return events

    def _invalid(self, frame: Frame) -> bool:
        bad = int(self.params["invalid_value"])
        return frame.rng == bad or frame.vel == bad

    def _on_absent(self, frame: Frame) -> list[dict]:
        if self.absent_since is None:
            self.absent_since = frame.t_ms
            return []
        if frame.t_ms - self.absent_since < self.params["absent_reset_ms"]:
            return []
        self._reset_track()
        event = self._emit(frame, "normal", "none", "no_human")
        return [event] if event is not None else []

    def _confirm_impact(self, frame: Frame) -> bool:
        if self.classifier is not None:
            return self._confirm_from_classifier(frame)
        return self._confirm_impact_rule(frame)

    def _confirm_from_classifier(self, frame: Frame) -> bool:
        predicted = self.classifier.predict(
            list(self.window),
            int(self.params["invalid_value"]),
            float(self.params["vel_fall"]),
        )
        if predicted is None:
            return False
        label, fall_p = predicted
        self.last_pred = label
        if label != "fall" or fall_p < float(self.params.get("fall_proba", 0.45)):
            return False
        bad = int(self.params["invalid_value"])
        valid = [item for item in self.window if item.rng != bad and item.vel != bad]
        if len(valid) >= 2:
            delta = valid[-1].rng - valid[0].rng
        else:
            delta = -1 if frame.vel < 0 else 1
        self.impact_on = True
        self.impact_sign = 1 if delta >= 0 else -1
        self.rng_before = valid[0].rng if valid else frame.rng
        self.rng_far = valid[-1].rng if valid else frame.rng
        self.impact_peak = max((abs(item.vel) for item in valid), default=abs(frame.vel))
        self.hgt_before = self._height_before(valid[0].t_ms) if valid else frame.hgt
        return True

    def _confirm_impact_rule(self, frame: Frame) -> bool:
        span = self.params["impact_span_ms"]
        high = [
            item
            for item in self.recent
            if frame.t_ms - item[0] <= span and abs(item[1]) >= self.params["vel_fall"]
        ]
        if len(high) < self.params["min_impact_frames"]:
            return False
        delta = high[-1][2] - high[0][2]
        vel_sum = sum(item[1] for item in high)
        if delta * vel_sum <= 0 or abs(delta) < self.params["impact_rng"]:
            return False
        self.impact_on = True
        self.impact_sign = 1 if delta > 0 else -1
        self.rng_before = high[0][2]
        self.rng_far = high[-1][2]
        self.impact_peak = max(abs(item[1]) for item in high)
        self.hgt_before = self._height_before(high[0][0])
        return True

    def _height_before(self, first_high_t: int) -> int | None:
        prior = [item[3] for item in self.recent if item[0] < first_high_t and item[3] is not None]
        if prior:
            return prior[-1]
        for item in self.recent:
            if item[0] >= first_high_t and item[3] is not None:
                return item[3]
        return None

    def _classifier_cancels(self, frame: Frame) -> bool:
        if self.classifier is None:
            return False
        predicted = self.classifier.predict(
            list(self.window),
            int(self.params["invalid_value"]),
            float(self.params["vel_fall"]),
        )
        if predicted is None:
            return False
        label, _ = predicted
        self.last_pred = label
        if label in ("stand", "walk", "sit", "bend"):
            self.cancel_windows += 1
            return self.cancel_windows >= int(self.params.get("min_cancel_windows", 3))
        self.cancel_windows = 0
        return False

    def _settle(self, frame: Frame) -> list[dict]:
        if self.paused or self.last_t is None:
            dt = 0
            self.paused = False
        else:
            dt = max(0, frame.t_ms - self.last_t)
        self.last_t = frame.t_ms

        further = (frame.rng - self.rng_far) * self.impact_sign
        if further > 0:
            self.rng_far = frame.rng

        if abs(frame.rng - self.rng_before) <= self.params["return_rng"]:
            return self._cancel(frame)

        retreat = (self.rng_far - frame.rng) * self.impact_sign
        if retreat >= self.params["rebound_rng"]:
            return self._cancel(frame)

        if self._classifier_cancels(frame):
            return self._cancel(frame)

        speed = abs(frame.vel)
        if further > 0 and speed > self.params["still_vel"]:
            self.still_acc = 0
            self.recover_high = 0
            self.posture_bad = 0
            return []

        if speed <= self.params["still_vel"]:
            self.motion_acc = 0
            self.recover_high = 0
            if self.hgt_before is not None and frame.hgt is None:
                return []
            self.still_acc += dt
            if self._posture_blocks(frame):
                self.posture_bad += 1
                held = self.still_acc >= self.params["still_suspect_ms"]
                if held and self.posture_bad >= self.params["min_impact_frames"]:
                    return self._cancel(frame)
                return []
            self.posture_bad = 0
            return self._advance_still(frame)

        self.posture_bad = 0
        if speed >= self.params["recover_vel"]:
            self.still_acc = 0
            self.recover_high += 1
            self.motion_acc += dt
            suspect_moving = self.phase == "suspect" and self.recover_high >= self.params["min_impact_frames"]
            if suspect_moving or self.motion_acc >= self.params["motion_cancel_ms"]:
                return self._cancel(frame)
        else:
            self.recover_high = 0
        return []

    def _posture_blocks(self, frame: Frame) -> bool:
        """True when height is known and this is not a drop to the floor.

        Missing hgt does not block. Board logs still have no height column.
        """
        if self.hgt_before is None or frame.hgt is None:
            return False
        if self.hgt_before - frame.hgt < self.params["height_drop"]:
            return True
        if frame.hgt > self.params["height_lie_max"]:
            return True
        return False

    def _advance_still(self, frame: Frame) -> list[dict]:
        events: list[dict] = []
        if self.phase == "normal" and self.still_acc >= self.params["still_suspect_ms"]:
            event = self._emit(frame, "suspect", "fall", "fast_then_still")
            if event is not None:
                events.append(event)
        if self.phase == "suspect" and self.still_acc >= self.params["still_alarm_ms"]:
            event = self._emit(frame, "alarm", "fall", "fast_then_still")
            if event is not None:
                events.append(event)
        return events

    def _cancel(self, frame: Frame) -> list[dict]:
        was_suspect = self.phase == "suspect"
        self._reset_track()
        if not was_suspect:
            return []
        event = self._emit(frame, "normal", "none", "recovered")
        return [event] if event is not None else []

    def _emit(self, frame: Frame, state: str, action: str, reason: str, force: bool = False) -> dict | None:
        key = (state, reason)
        if not force and key == self.last_emit:
            return None
        if (
            not force
            and state == "normal"
            and self.phase == "normal"
            and self.last_emit is not None
            and self.last_emit[0] == "normal"
        ):
            return None
        self.phase = state
        self.last_emit = key
        return make_event(self.params, frame, state, action, reason)


def apply_uplink(detector: FallDetector, message: dict, frame: Frame) -> dict:
    """Accept the uplink line documented for classmate B.

    The only command is ``{"device_id": "...", "cmd": "clear"}``.
    """
    if not isinstance(message, dict):
        raise CommandError("命令必须是 JSON 对象")
    if message.get("cmd") != "clear":
        raise CommandError(f"未知命令: {message.get('cmd')}")
    device = message.get("device_id")
    if device != detector.params["device_id"]:
        raise CommandError("device_id 与本机不一致")
    return detector.clear(frame)


def apply_uplink_text(detector: FallDetector, line: str, frame: Frame) -> dict:
    """Parse one uplink JSON line and apply it."""
    stripped = line.strip()
    if not stripped:
        raise CommandError("命令是空的")
    try:
        message = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise CommandError("命令不是合法 JSON") from exc
    return apply_uplink(detector, message, frame)
