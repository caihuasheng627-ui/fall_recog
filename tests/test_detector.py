"""Unit tests for gaps the synthetic scorecard does not cover."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fall_recog.evaluate import _peak, classify_events, outcome_ok, render_report, score_file
from fall_recog.rules import CommandError, FallDetector, apply_uplink, apply_uplink_text, load_params
from fall_recog.schema import Frame

ROOT = Path(__file__).resolve().parents[1]
DT = 50


def load() -> dict:
    params = load_params(ROOT / "params.json")
    params["use_classifier"] = False
    return params


def make_frame(
    t_ms: int,
    frame_id: int,
    rng: int,
    vel: int,
    human: int = 1,
    hgt: int | None = 150,
    label: str = "fall",
) -> Frame:
    return Frame(
        t_ms=t_ms,
        frame_id=frame_id,
        gear=1 if human else 0,
        rng=rng,
        vel=vel,
        snr=20.0,
        human=human,
        lb=100 if human else 0,
        ub=200 if human else 0,
        label=label,
        hgt=hgt,
    )


class Script:
    def __init__(self) -> None:
        self.t_ms = 0
        self.frame_id = 0
        self.frames: list[Frame] = []

    def add(
        self,
        ms: int,
        rng: float,
        vel: int,
        human: int = 1,
        hgt: int | None = 150,
        label: str = "fall",
        rng_end: float | None = None,
    ) -> None:
        steps = ms // DT
        end = rng if rng_end is None else rng_end
        for index in range(steps):
            if steps == 1:
                current = end
            else:
                current = rng + (end - rng) * index / (steps - 1)
            self.frames.append(
                make_frame(self.t_ms, self.frame_id, int(round(current)), vel, human, hgt, label)
            )
            self.t_ms += DT
            self.frame_id += 1


def run_frames(frames: list[Frame], params: dict | None = None) -> tuple[FallDetector, list[dict]]:
    detector = FallDetector(params or load())
    events: list[dict] = []
    for frame in frames:
        events.extend(detector.push(frame))
    return detector, events


def states(events: list[dict]) -> list[str]:
    return [event["state"] for event in events]


class ClockTests(unittest.TestCase):
    def test_human_gap_does_not_jump_stillness(self) -> None:
        params = load()
        detector, script = self._armed(hgt=25)
        events: list[dict] = []
        while detector.still_acc < 2600:
            script.add(DT, detector.rng_far, 0, hgt=25)
            events.extend(detector.push(script.frames[-1]))
        self.assertLess(detector.still_acc, params["still_suspect_ms"])
        self.assertEqual(detector.phase, "normal")
        frozen = detector.still_acc
        self._gap(detector, script, 600, detector.rng_far)
        script.add(DT, detector.rng_far, 0, hgt=25)
        events.extend(detector.push(script.frames[-1]))
        self.assertEqual(detector.still_acc, frozen)
        self.assertEqual(detector.phase, "normal")
        self.assertNotIn("suspect", states(events))

    def test_human_gap_does_not_cancel_impact_early(self) -> None:
        params = load()
        detector, script = self._armed(hgt=25)
        while detector.motion_acc < 1600:
            script.add(DT, detector.rng_far, 50, hgt=25)
            detector.push(script.frames[-1])
        self.assertTrue(detector.impact_on)
        self.assertLess(detector.motion_acc, params["motion_cancel_ms"])
        frozen = detector.motion_acc
        self._gap(detector, script, 600, detector.rng_far)
        script.add(DT, detector.rng_far, 50, hgt=25)
        detector.push(script.frames[-1])
        self.assertTrue(detector.impact_on)
        self.assertEqual(detector.motion_acc, frozen)

    def test_invalid_frame_still_pauses(self) -> None:
        detector, script = self._armed(hgt=25)
        while detector.still_acc < 2600:
            script.add(DT, detector.rng_far, 0, hgt=25)
            detector.push(script.frames[-1])
        frozen = detector.still_acc
        script.frames.append(
            make_frame(script.t_ms, script.frame_id, -999, -999, human=1, hgt=25)
        )
        script.t_ms += 400
        script.frame_id += 1
        detector.push(script.frames[-1])
        script.add(DT, detector.rng_far, 0, hgt=25)
        detector.push(script.frames[-1])
        self.assertEqual(detector.still_acc, frozen)
        self.assertTrue(detector.impact_on)

    def _armed(self, hgt: int | None) -> tuple[FallDetector, Script]:
        script = Script()
        script.add(1000, 150, 0, hgt=150)
        script.add(400, 150, -120, hgt=hgt, rng_end=80)
        detector, _ = run_frames(script.frames)
        self.assertTrue(detector.impact_on)
        return detector, script

    def _gap(self, detector: FallDetector, script: Script, ms: int, rng: int) -> None:
        for _ in range(ms // DT):
            script.frames.append(
                make_frame(script.t_ms, script.frame_id, rng, 0, human=0, hgt=None)
            )
            detector.push(script.frames[-1])
            script.t_ms += DT
            script.frame_id += 1


class RuleSeparationTests(unittest.TestCase):
    def test_thresholds_stay_put(self) -> None:
        params = load()
        self.assertEqual(params["vel_fall"], 90)
        self.assertEqual(params["impact_rng"], 30)
        self.assertEqual(params["still_alarm_ms"], 8000)

    def test_fast_sit_with_height_does_not_alarm(self) -> None:
        _, events = run_frames(self._fast_then_still(hgt_after=85))
        self.assertNotIn("alarm", states(events))
        self.assertNotIn("suspect", states(events))

    def test_same_sit_without_height_still_alarms(self) -> None:
        frames = [replace(frame, hgt=None) for frame in self._fast_then_still(hgt_after=85)]
        _, events = run_frames(frames)
        self.assertIn("alarm", states(events))
        self.assertEqual(classify_events(events), "alarm")

    def test_stop_at_standing_height_does_not_alarm(self) -> None:
        _, events = run_frames(self._fast_then_still(hgt_after=150, rush_hgt=150))
        self.assertNotIn("alarm", states(events))

    def test_same_stop_without_height_alarms(self) -> None:
        frames = self._fast_then_still(hgt_after=150, rush_hgt=150)
        _, events = run_frames([replace(frame, hgt=None) for frame in frames])
        self.assertEqual(classify_events(events), "alarm")

    def test_rebound_cancels_without_height(self) -> None:
        params = load()
        script = Script()
        script.add(1000, 150, 0, hgt=None)
        script.add(400, 150, -140, hgt=None, rng_end=80)
        detector = FallDetector(params)
        for frame in script.frames:
            detector.push(frame)
        self.assertTrue(detector.impact_on)
        target = detector.rng_far + params["rebound_rng"] + 5
        self.assertGreater(abs(target - detector.rng_before), params["return_rng"])
        cleared_rng = None
        script.add(500, detector.rng_far, 40, hgt=None, rng_end=target)
        for frame in script.frames[-500 // DT :]:
            was_on = detector.impact_on
            detector.push(frame)
            if was_on and not detector.impact_on:
                cleared_rng = frame.rng
                break
        self.assertIsNotNone(cleared_rng)
        assert cleared_rng is not None
        self.assertGreater(abs(cleared_rng - 150), params["return_rng"])
        script.add(10000, target, 0, hgt=None)
        events: list[dict] = []
        for frame in script.frames[-10000 // DT :]:
            events.extend(detector.push(frame))
        self.assertNotIn("alarm", states(events))
        self.assertFalse(detector.impact_on)

    def test_slow_fall_stays_a_miss(self) -> None:
        script = Script()
        script.add(1000, 150, 0, hgt=150)
        script.add(2500, 150, -40, hgt=25, rng_end=70)
        script.add(10000, 70, 0, hgt=25)
        _, events = run_frames(script.frames)
        self.assertNotIn("alarm", states(events))

    def test_fast_fall_with_lying_height_alarms(self) -> None:
        _, events = run_frames(self._fast_then_still(hgt_after=25))
        self.assertEqual(classify_events(events), "alarm")

    def test_still_just_outside_return_rng_alarms(self) -> None:
        params = load()
        script = Script()
        script.add(1000, 150, 0, hgt=150)
        script.add(400, 150, -120, hgt=25, rng_end=80)
        detector = FallDetector(params)
        for frame in script.frames:
            detector.push(frame)
        self.assertTrue(detector.impact_on)
        offset = params["return_rng"] + 2
        self.assertLess(offset, params["rebound_rng"])
        target = detector.rng_far - detector.impact_sign * offset
        script.add(10000, target, 0, hgt=25)
        for frame in script.frames[-10000 // DT :]:
            detector.push(frame)
        self.assertEqual(detector.phase, "alarm")

    def test_continued_fall_then_still_alarms(self) -> None:
        params = load()
        script = Script()
        script.add(1000, 150, 0, hgt=150)
        script.add(400, 150, -120, hgt=25, rng_end=80)
        detector = FallDetector(params)
        for frame in script.frames:
            detector.push(frame)
        self.assertTrue(detector.impact_on)
        far = detector.rng_far + detector.impact_sign * 50
        script.add(params["motion_cancel_ms"] + 500, detector.rng_far, -80, hgt=25, rng_end=far)
        for frame in script.frames[-(params["motion_cancel_ms"] + 500) // DT :]:
            detector.push(frame)
        self.assertTrue(detector.impact_on)
        script.add(10000, far, 0, hgt=25)
        for frame in script.frames[-10000 // DT :]:
            detector.push(frame)
        self.assertEqual(detector.phase, "alarm")

    def test_twitch_pauses_stillness_but_does_not_cancel(self) -> None:
        params = load()
        script = Script()
        script.add(1000, 150, 0, hgt=150)
        script.add(400, 150, -120, hgt=25, rng_end=80)
        detector = FallDetector(params)
        for frame in script.frames:
            detector.push(frame)
        while detector.still_acc < 2600:
            script.add(DT, detector.rng_far, 0, hgt=25)
            detector.push(script.frames[-1])
        frozen = detector.still_acc
        twitch = (params["still_vel"] + params["recover_vel"]) // 2
        for _ in range(6):
            script.add(DT, detector.rng_far, twitch, hgt=25)
            detector.push(script.frames[-1])
        self.assertTrue(detector.impact_on)
        self.assertEqual(detector.still_acc, frozen)
        self.assertEqual(detector.phase, "normal")
        while detector.phase != "alarm":
            script.add(DT, detector.rng_far, 0, hgt=25)
            detector.push(script.frames[-1])
            if script.t_ms > 40000:
                self.fail("twitch cancelled the impact or stalled stillness")
        self.assertEqual(detector.phase, "alarm")

    def test_dropout_during_impact_does_not_reset(self) -> None:
        params = load()
        script = Script()
        script.add(1000, 150, 0, hgt=150)
        script.add(400, 150, -120, hgt=25, rng_end=80)
        detector = FallDetector(params)
        for frame in script.frames:
            detector.push(frame)
        self.assertTrue(detector.impact_on)
        for _ in range(params["absent_reset_ms"] // DT + 4):
            script.frames.append(make_frame(script.t_ms, script.frame_id, 80, 0, human=0, hgt=None))
            detector.push(script.frames[-1])
            script.t_ms += DT
            script.frame_id += 1
        self.assertTrue(detector.impact_on)
        self.assertEqual(detector.phase, "normal")
        script.add(10000, detector.rng_far, 0, hgt=25)
        for frame in script.frames[-10000 // DT :]:
            detector.push(frame)
        self.assertEqual(detector.phase, "alarm")

    def _fast_then_still(self, hgt_after: int, rush_hgt: int | None = None) -> list[Frame]:
        script = Script()
        script.add(1000, 150, 0, hgt=150)
        script.add(400, 150, -120, hgt=150 if rush_hgt is None else rush_hgt, rng_end=80)
        script.add(10000, 80, 0, hgt=hgt_after)
        return script.frames


class EvaluationTests(unittest.TestCase):
    def test_final_state_not_history(self) -> None:
        self.assertEqual(
            classify_events([{"state": "normal"}, {"state": "suspect"}, {"state": "normal"}]),
            "normal",
        )
        self.assertEqual(
            classify_events([{"state": "normal"}, {"state": "alarm"}, {"state": "normal"}]),
            "normal",
        )
        self.assertFalse(outcome_ok("normal", "alarm"))
        self.assertEqual(classify_events([{"state": "suspect"}]), "suspect")
        self.assertEqual(classify_events([{"state": "alarm"}]), "alarm")

    def test_stand_up_after_suspect_is_normal(self) -> None:
        script = Script()
        script.add(1000, 150, 0, hgt=150)
        script.add(400, 150, -120, hgt=25, rng_end=80)
        detector = FallDetector(load())
        events: list[dict] = []
        for frame in script.frames:
            events.extend(detector.push(frame))
        guard = 0
        while detector.phase != "suspect" and guard < 200:
            script.add(DT, detector.rng_far, 0, hgt=25)
            events.extend(detector.push(script.frames[-1]))
            guard += 1
        self.assertEqual(detector.phase, "suspect")
        for _ in range(4):
            script.add(DT, detector.rng_far, 60, hgt=25)
            events.extend(detector.push(script.frames[-1]))
        self.assertEqual(events[-1]["reason"], "recovered")
        self.assertEqual(classify_events(events), "normal")
        self.assertIn("suspect", states(events))

    def test_alarm_then_no_human_stays_alarm(self) -> None:
        detector = FallDetector(load())
        played: list[dict] = []
        script = Script()
        script.add(1000, 150, 0, hgt=150)
        script.add(400, 150, -120, hgt=25, rng_end=80)
        script.add(10000, 80, 0, hgt=25)
        for frame in script.frames:
            played.extend(detector.push(frame))
        self.assertEqual(detector.phase, "alarm")
        for _ in range(1000 // DT):
            script.frames.append(make_frame(script.t_ms, script.frame_id, 80, 0, human=0, hgt=None))
            played.extend(detector.push(script.frames[-1]))
            script.t_ms += DT
            script.frame_id += 1
        self.assertEqual(detector.phase, "alarm")
        self.assertNotIn("no_human", [event["reason"] for event in played])
        self.assertEqual(classify_events(played), "alarm")

    def test_peak_ignores_invalid(self) -> None:
        params = load()
        frames = [
            make_frame(0, 0, 150, 40),
            make_frame(50, 1, 150, -999),
            make_frame(100, 2, -999, 800),
            make_frame(150, 3, 140, 10),
        ]
        self.assertEqual(_peak(frames, params["invalid_value"]), 40)
        self.assertEqual(_peak([make_frame(0, 0, -999, -999)], params["invalid_value"]), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "P09_fall_01_test.csv"
            path.write_text(
                "\n".join(
                    [
                        "t_ms,frame_id,gear,rng,vel,snr,human,lb,ub,label",
                        "0,0,1,150,40,20.0,1,100,200,fall",
                        "50,1,1,150,-999,20.0,1,100,200,fall",
                        "100,2,1,-999,800,20.0,1,100,200,fall",
                        "150,3,1,140,10,20.0,1,100,200,fall",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            result = score_file(path, params)
        self.assertEqual(result["peak"], 40)

    def test_report_uses_final_state_wording(self) -> None:
        item = {
            "label": "stand",
            "kind": "normal",
            "ok": True,
            "peak": 10,
            "path": Path("P01_stand_01_test.csv"),
            "onset": None,
            "alarm_at": None,
            "delay": None,
            "expect": "no_alarm",
            "note": "",
            "warnings": [],
        }
        text = render_report("t", [item], load(), "note")
        self.assertIn("结束时", text)
        self.assertNotIn("有没有进入", text)


class UplinkTests(unittest.TestCase):
    def test_clear_line_resets_and_does_not_realarm(self) -> None:
        params = load()
        detector, events = run_frames(RuleSeparationTests()._fast_then_still(hgt_after=25), params)
        self.assertEqual(detector.phase, "alarm")
        frame = events[-1]
        current = make_frame(frame["t_ms"], frame["frame_id"], frame["rng"], frame["vel"], hgt=25)
        with self.assertRaises(CommandError):
            apply_uplink(detector, {"device_id": "other", "cmd": "clear"}, current)
        self.assertEqual(detector.phase, "alarm")
        with self.assertRaises(CommandError):
            apply_uplink_text(detector, '{"device_id":"ws63-1","cmd":"reset"}', current)
        self.assertEqual(detector.phase, "alarm")
        cleared = apply_uplink_text(
            detector,
            ' {"device_id":"ws63-1","cmd":"clear"}\n',
            current,
        )
        self.assertEqual(cleared["state"], "normal")
        self.assertEqual(cleared["reason"], "cleared")
        self.assertEqual(detector.phase, "normal")
        later: list[dict] = []
        for step in range(1, 40):
            later.extend(
                detector.push(
                    make_frame(
                        current.t_ms + step * DT,
                        current.frame_id + step,
                        current.rng,
                        1,
                        hgt=25,
                    )
                )
            )
        self.assertNotIn("alarm", states(later))

    def test_bad_lines_do_not_clear(self) -> None:
        detector = FallDetector(load())
        frame = make_frame(0, 0, 80, 0, hgt=25)
        detector.started = True
        detector.phase = "alarm"
        detector.impact_on = True
        for line in ("", "   ", "{", "[]", "null"):
            with self.assertRaises(CommandError):
                apply_uplink_text(detector, line, frame)
            self.assertEqual(detector.phase, "alarm")
            self.assertTrue(detector.impact_on)


if __name__ == "__main__":
    unittest.main()
