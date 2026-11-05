from __future__ import annotations

import unittest

from fall_recog.features import FEATURE_NAMES, extract_features
from fall_recog.schema import IQ_LEN, Frame, synthetic_iq
from fall_recog.train import window_label


def frame(t_ms: int, rng: int, vel: int, human: int = 1) -> Frame:
    return Frame(
        t_ms=t_ms,
        frame_id=t_ms // 50,
        gear=1,
        rng=rng,
        vel=vel,
        snr=20.0,
        human=human,
        lb=100,
        ub=200,
        label="fall",
        iq=synthetic_iq(rng, vel, human, 20.0),
    )


class FeatureTests(unittest.TestCase):
    def test_iq_features_lead_and_length(self) -> None:
        self.assertTrue(FEATURE_NAMES[0].startswith("iq_mag_"))
        frames = [frame(i * 50, 150 - i, -40) for i in range(10)]
        feats = extract_features(frames, -999, 90)
        self.assertIsNotNone(feats)
        assert feats is not None
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        self.assertGreater(feats["iq_energy"], 0)
        self.assertNotIn("hgt", FEATURE_NAMES)

    def test_synthetic_iq_is_40(self) -> None:
        self.assertEqual(len(synthetic_iq(150, -10, 1, 20.0)), IQ_LEN)
        self.assertEqual(sum(synthetic_iq(0, 0, 0, 0.0)), 0)


class LabelTests(unittest.TestCase):
    def test_fall_core_then_lie(self) -> None:
        self.assertEqual(window_label("fall", 2500, 3000), "stand")
        self.assertEqual(window_label("fall", 3500, 3000), "fall")
        self.assertEqual(window_label("fall", 8000, 3000), "lie")
        self.assertEqual(window_label("sit", 8000, 3000), "sit")
        self.assertEqual(window_label("lie", 8000, 3000), "lie")


if __name__ == "__main__":
    unittest.main()
