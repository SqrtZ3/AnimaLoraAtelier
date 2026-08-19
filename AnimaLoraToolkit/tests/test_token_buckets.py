import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.token_buckets import generate_token_buckets


class TestGenerateTokenBuckets(unittest.TestCase):
    CFG = dict(token_counts=[4032, 4200], max_aspect_ratio=2.0,
               patch_pixels=16, min_dim_px=512, max_dim_px=2016)

    def test_all_buckets_hit_exact_token_count(self):
        bks = generate_token_buckets(**self.CFG)
        self.assertTrue(bks)
        for w, h in bks:
            self.assertIn((w // 16) * (h // 16), (4032, 4200))
            self.assertLessEqual(max(w / h, h / w), 2.0 + 1e-9)
            self.assertTrue(512 <= w <= 2016 and 512 <= h <= 2016)

    def test_near_square_present_for_4032(self):
        bks = generate_token_buckets([4032], 2.0, 16, 512, 2016)
        self.assertIn((1008, 1024), bks)

    def test_n1_single_token_count(self):
        bks = generate_token_buckets([4032], 2.0, 16, 512, 2016)
        self.assertEqual({(w // 16) * (h // 16) for w, h in bks}, {4032})

    def test_deterministic_sorted(self):
        bks = generate_token_buckets(**self.CFG)
        self.assertEqual(
            bks,
            sorted(bks, key=lambda wh: ((wh[0] // 16) * (wh[1] // 16), wh[0], wh[1])),
        )

    def test_matches_golden_fixture(self):
        fx = pathlib.Path(__file__).resolve().parent / "fixtures" / "token_buckets_canonical.json"
        expect = [tuple(x) for x in json.loads(fx.read_text())]
        got = generate_token_buckets(**self.CFG)
        self.assertEqual(got, expect)


if __name__ == "__main__":
    unittest.main()
