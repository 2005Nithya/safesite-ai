import unittest

import numpy as np

import detection
from detection import _prepare_frame_for_inference, evaluate_person_ppe, process_frame


class DetectionLogicTests(unittest.TestCase):
    def test_safe_when_helmet_and_vest_overlap_person(self):
        helmet_found, vest_found = evaluate_person_ppe(
            (0, 0, 200, 300),
            [(40, 40, 90, 100)],
            [],
            [(60, 180, 120, 240)],
        )
        self.assertTrue(helmet_found)
        self.assertTrue(vest_found)

    def test_missing_helmet_when_only_vest_is_present(self):
        helmet_found, vest_found = evaluate_person_ppe(
            (0, 0, 200, 300),
            [],
            [],
            [(60, 180, 120, 240)],
        )
        self.assertFalse(helmet_found)
        self.assertTrue(vest_found)

    def test_nonhelmet_box_overrides_helmet(self):
        helmet_found, vest_found = evaluate_person_ppe(
            (0, 0, 200, 300),
            [(40, 40, 90, 100)],
            [(50, 60, 100, 110)],
            [],
        )
        self.assertFalse(helmet_found)
        self.assertFalse(vest_found)

    def test_prepare_frame_for_inference_resizes_large_frames(self):
        frame = np.zeros((1200, 1600, 3), dtype=np.uint8)
        prepared = _prepare_frame_for_inference(frame, target_size=320)
        self.assertEqual(prepared.shape[0], 240)
        self.assertEqual(prepared.shape[1], 320)

    def test_process_frame_falls_back_when_model_is_unavailable(self):
        original_model = detection._model
        detection._model = None
        detection._get_model = lambda: None
        try:
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            processed, workers, safe_workers, violations = process_frame(frame)
            self.assertEqual(processed.shape, frame.shape)
            self.assertEqual((workers, safe_workers, violations), (0, 0, 0))
        finally:
            detection._model = original_model
            detection._get_model = lambda: detection._model


if __name__ == "__main__":
    unittest.main()
