import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


class FakeRecord:
    query_sequence = "CAAACA"

    def get_tag(self, tag):
        if tag == "MM":
            return "A+a.,1,0;"
        if tag == "ML":
            return [128, 255]
        raise KeyError(tag)


class ModbamGoldSitesTest(unittest.TestCase):
    def test_parse_bed_gold_uses_zero_based_site_key(self):
        from validate.evaluate_modbam_gold_sites import load_gold_sites

        with tempfile.TemporaryDirectory() as tmpdir:
            bed = Path(tmpdir) / "gold.bed"
            bed.write_text("chr1\t9\t10\tm6A_1\t3\t+\n", encoding="utf-8")
            gold = load_gold_sites(bed, gold_format="bed")

        self.assertIn(("chr1", 9, "+"), gold)
        self.assertEqual(gold[("chr1", 9, "+")]["name"], "m6A_1")

    def test_fallback_mm_ml_parser_returns_query_probabilities(self):
        from validate.evaluate_modbam_gold_sites import fallback_parse_mm_ml

        calls = fallback_parse_mm_ml(FakeRecord(), canonical_base="A", mod_code="a")

        self.assertEqual(set(calls), {2, 3})
        self.assertAlmostEqual(calls[2], 128 / 255)
        self.assertEqual(calls[3], 1.0)

    @unittest.skipUnless(find_spec("sklearn") is not None, "scikit-learn is not installed")
    def test_site_rows_and_metrics(self):
        from validate.evaluate_modbam_gold_sites import (
            SiteStats,
            build_site_rows,
            compute_metrics,
        )

        pos = SiteStats()
        pos.add_coverage(1.0)
        pos.add_coverage(None)
        neg = SiteStats()
        neg.add_coverage(None)
        neg.add_coverage(None)
        stats = {
            ("chr1", 9, "+"): pos,
            ("chr1", 19, "+"): neg,
        }
        gold = {
            ("chr1", 9, "+"): {
                "name": "gold1",
                "support": 2,
            }
        }

        rows = build_site_rows(
            stats,
            gold,
            min_coverage=1,
            score_column="mod_fraction",
            threshold=0.5,
        )
        metrics = compute_metrics(rows, threshold=0.5)

        self.assertEqual(len(rows), 2)
        self.assertEqual(metrics["num_positive"], 1)
        self.assertEqual(metrics["num_negative"], 1)
        self.assertEqual(metrics["threshold_metrics"]["tp"], 1)
        self.assertEqual(metrics["threshold_metrics"]["tn"], 1)
        self.assertEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
