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

    def test_parse_chr_pos_ratio_tsv_uses_ratio_as_support(self):
        from validate.evaluate_modbam_gold_sites import load_gold_sites

        with tempfile.TemporaryDirectory() as tmpdir:
            table = Path(tmpdir) / "gold.tsv"
            table.write_text(
                "chr\tpos\tstrand\tcontext\tratio\n"
                "chr1\t14517\t-\tGGACT\t0.4145\n",
                encoding="utf-8",
            )
            gold = load_gold_sites(table, gold_format="auto")

        self.assertIn(("chr1", 14516, "-"), gold)
        self.assertAlmostEqual(gold[("chr1", 14516, "-")]["support"], 0.4145)

    def test_fallback_mm_ml_parser_returns_query_probabilities(self):
        from validate.evaluate_modbam_gold_sites import fallback_parse_mm_ml

        calls = fallback_parse_mm_ml(FakeRecord(), canonical_base="A", mod_code="a")

        self.assertEqual(set(calls), {2, 3})
        self.assertAlmostEqual(calls[2], 128 / 255)
        self.assertEqual(calls[3], 1.0)

    def test_coordinate_convention_transform_applies_shift_and_strand(self):
        from validate.check_gold_coordinate_conventions import transform_gold

        gold = {
            ("chr1", 10, "+"): {"name": "plus"},
            ("chr1", 20, "-"): {"name": "minus"},
        }

        shifted = transform_gold(gold, shift=-1, convention="as_is")
        self.assertIn(("chr1", 9, "+"), shifted)
        self.assertIn(("chr1", 19, "-"), shifted)

        flipped = transform_gold(gold, shift=0, convention="flip_gold_strand")
        self.assertIn(("chr1", 10, "-"), flipped)
        self.assertIn(("chr1", 20, "+"), flipped)

        ignored = transform_gold(gold, shift=1, convention="ignore_strand")
        self.assertIn(("chr1", 11, "."), ignored)
        self.assertIn(("chr1", 21, "."), ignored)

    def test_collapse_stats_by_strand_merges_coverage_and_probs(self):
        from validate.check_gold_coordinate_conventions import collapse_stats_by_strand
        from validate.evaluate_modbam_gold_sites import SiteStats

        plus = SiteStats()
        plus.add_coverage(0.5)
        minus = SiteStats()
        minus.add_coverage(None)
        stats = {
            ("chr1", 10, "+"): plus,
            ("chr1", 10, "-"): minus,
        }

        collapsed = collapse_stats_by_strand(stats)

        self.assertEqual(set(collapsed), {("chr1", 10, ".")})
        self.assertEqual(collapsed[("chr1", 10, ".")].coverage, 2)
        self.assertEqual(collapsed[("chr1", 10, ".")].mod_probs, [0.5])

    def test_negative_control_false_positive_summary(self):
        from validate.evaluate_modbam_gold_sites import SiteStats
        from validate.evaluate_modbam_negative_control import (
            build_negative_rows,
            threshold_false_positive_rows,
        )

        low = SiteStats()
        low.add_coverage(0.1)
        low.add_coverage(None)
        high = SiteStats()
        high.add_coverage(0.9)
        high.add_coverage(0.9)
        stats = {
            ("chr1", 10, "+"): low,
            ("chr1", 20, "+"): high,
        }

        rows = build_negative_rows(
            stats,
            min_coverage=1,
            score_column="mean_prob_zero_filled",
            show_progress=False,
        )
        sweep = threshold_false_positive_rows(rows, [0.5])

        self.assertEqual(len(rows), 2)
        self.assertEqual(sweep[0]["false_positive_sites"], 1)
        self.assertEqual(sweep[0]["false_positive_fraction"], 0.5)

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
        self.assertEqual(metrics["threshold_metrics"]["fpr"], 0.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertIn("0.01", metrics["tpr_at_fpr"])


if __name__ == "__main__":
    unittest.main()
