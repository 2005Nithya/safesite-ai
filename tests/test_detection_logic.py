import unittest

from detection import evaluate_person_ppe


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


if __name__ == "__main__":
    unittest.main()
