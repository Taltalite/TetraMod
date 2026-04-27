import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


class TetramodCliSmokeTest(unittest.TestCase):
    def test_cli_builds_expected_commands(self):
        from tetramod.cli import build_parser

        parser = build_parser()
        actions = [action for action in parser._actions if action.dest == "command"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(set(actions[0].choices), {"train", "train_promote", "basecaller"})

    def test_subcommand_help_parses_without_bonito_runtime_imports(self):
        from tetramod.cli import main

        for command in ("train", "train_promote", "basecaller"):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    main([command, "--help"])
            self.assertEqual(ctx.exception.code, 0)

    def test_train_promote_defaults_to_a_head(self):
        from tetramod.cli.train_promote import PROMOTE_HEAD_BASE, argparser

        args = argparser().parse_args(["workdir", "--pretrained", "model_dir"])
        self.assertEqual(args.promote_base, PROMOTE_HEAD_BASE)
        self.assertIsNone(args.promote_stage)

        args = argparser().parse_args(
            [
                "workdir",
                "--pretrained",
                "model_dir",
                "--promote-stage",
                "llp",
                "--llp-proportion",
                "25",
                "--llp-bag-size",
                "8",
            ]
        )
        self.assertEqual(args.promote_stage, "llp")
        self.assertEqual(args.llp_proportion, 25.0)
        self.assertEqual(args.llp_bag_size, 8)

    def test_train_promote_restricts_model_to_a_head(self):
        from tetramod.cli.train_promote import prepare_promote_config

        config = {
            "model": {
                "mod_bases": ["A", "C", "G", "T"],
                "mod_global_labels": [
                    "canonical_A",
                    "canonical_C",
                    "canonical_G",
                    "canonical_T",
                    "m6A",
                ],
                "mod_head_defs": {
                    "A": ["canonical_A", "m6A"],
                    "C": ["canonical_C"],
                    "G": ["canonical_G"],
                    "T": ["canonical_T"],
                },
            }
        }

        promoted = prepare_promote_config(config)

        self.assertEqual(promoted["model"]["mod_bases"], ["A"])
        self.assertEqual(promoted["model"]["mod_head_defs"], {"A": ["canonical_A", "m6A"]})

    def test_train_promote_control_stage_resolution_and_loss_path(self):
        from tetramod.training_promote import (
            CONTROL_WARMUP_LOSS_PATH,
            LLP_LOSS_PATH,
            PROMOTE_STAGE_CONTROL,
            PROMOTE_STAGE_LLP,
            normalize_llp_proportion,
            resolve_llp_settings,
            resolve_promote_stage,
        )

        config = {"training": {"promote_stage": "control"}}

        self.assertEqual(resolve_promote_stage(config), PROMOTE_STAGE_CONTROL)
        self.assertEqual(resolve_promote_stage({}, "control"), PROMOTE_STAGE_CONTROL)
        self.assertEqual(resolve_promote_stage({}, "llp"), PROMOTE_STAGE_LLP)
        self.assertEqual(CONTROL_WARMUP_LOSS_PATH, "a_head_control_warmup_viterbi_bce")
        self.assertEqual(LLP_LOSS_PATH, "a_head_llp_mean_pool_proportion_bce")
        self.assertEqual(normalize_llp_proportion(25), 0.25)
        self.assertEqual(
            resolve_llp_settings({"training": {"llp_proportion": 0.5, "llp_bag_size": 4}}),
            {"llp_proportion": 0.5, "llp_bag_size": 4},
        )
        self.assertEqual(resolve_llp_settings({"training": {"llp_bag_size": 4}}), {"llp_bag_size": 4})

    def test_promote_llp_loss_mean_pools_reads_then_bags(self):
        import torch

        from tetramod.training_promote import LLPProportionLoss

        class FakeModel:
            mod_bases = ["A"]
            mod_head_defs = {"A": ["canonical_A", "m6A"]}
            standalone_mod_head = True
            mod_loss_weight = 2.0

            def __init__(self, logits):
                self.logits = logits

            def align_predictions_to_targets(self, outputs, targets, target_lengths, mod_targets):
                return {
                    "per_head": {
                        "A": {
                            "flat_logits": self.logits,
                            "flat_targets": torch.zeros((3,), dtype=torch.long),
                            "flat_sample_indices": torch.tensor([0, 0, 1]),
                        }
                    }
                }

        logits = torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]], requires_grad=True)
        model = FakeModel(logits)
        criterion = LLPProportionLoss(model, llp_proportion=75, llp_bag_size=2)
        outputs = {
            "base_scores": torch.zeros((1, 2, 1)),
            "sample_keys": torch.tensor([0, 1]),
            "bag_keys": torch.tensor([9, 9]),
            "bag_targets": torch.tensor([0.75, 0.75]),
        }

        losses = criterion(
            outputs,
            targets=torch.zeros((2, 1), dtype=torch.long),
            target_lengths=torch.ones((2,), dtype=torch.long),
            mod_targets=torch.zeros((2, 1), dtype=torch.long),
        )

        read0 = torch.sigmoid(torch.tensor([0.0, 2.0])).mean()
        read1 = torch.sigmoid(torch.tensor([4.0]))
        bag_prob = torch.stack([read0, read1.squeeze(0)]).mean()
        expected = torch.nn.functional.binary_cross_entropy(
            bag_prob.clamp(1e-6, 1.0 - 1e-6),
            torch.tensor(0.75),
        )
        self.assertTrue(torch.allclose(losses["llp_loss"], expected))
        self.assertTrue(torch.allclose(losses["total_loss"], expected * 2.0))
        self.assertEqual(losses["llp_num_bags"].item(), 1.0)
        self.assertEqual(losses["llp_num_reads"].item(), 2.0)

    def test_synthetic_llp_controls_builder_smoke(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        import numpy as np

        def write_source(directory, source_offset):
            directory.mkdir(parents=True)
            n = 6
            np.save(directory / "chunks.npy", np.ones((n, 8), dtype=np.float16) * source_offset)
            np.save(directory / "references.npy", np.ones((n, 3), dtype=np.uint8))
            np.save(directory / "reference_lengths.npy", np.full((n,), 3, dtype=np.uint16))
            np.save(directory / "mod_targets.npy", np.zeros((n, 3), dtype=np.int16))
            np.savez(
                directory / "metadata.npz",
                record_id=np.asarray([f"read_{source_offset}_{idx}" for idx in range(n)], dtype=str),
                pod5_read_id=np.asarray([f"pod5_{source_offset}_{idx}" for idx in range(n)], dtype=str),
                run_id=np.asarray([f"run_{source_offset}"] * n, dtype=str),
                contig=np.asarray(["tx1"] * n, dtype=str),
                primary_site_key=np.asarray(["tx1:100:1:A"] * n, dtype=str),
                kmer_context=np.asarray(["GGACT"] * n, dtype=str),
                motif_context=np.asarray(["DRACH"] * n, dtype=str),
                ref_start=np.arange(n, dtype=np.int64),
                ref_end=np.arange(n, dtype=np.int64) + 3,
                ref_strand=np.ones((n,), dtype=np.int8),
                chunk_start=np.zeros((n,), dtype=np.int64),
                chunk_end=np.full((n,), 8, dtype=np.int64),
                primary_site_pos=np.full((n,), 100, dtype=np.int64),
                mean_qscore=np.full((n,), 12.0, dtype=np.float32),
                mapping_accuracy=np.full((n,), 0.99, dtype=np.float32),
                mapping_coverage=np.full((n,), 0.95, dtype=np.float32),
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full_dir = root / "full"
            canonical_dir = root / "canonical"
            out_dir = root / "out"
            write_source(full_dir, 100)
            write_source(canonical_dir, 0)

            subprocess.run(
                [
                    sys.executable,
                    "gen_data/build_synthetic_llp_from_controls.py",
                    "--full-mod-dataset",
                    str(full_dir),
                    "--canonical-dataset",
                    str(canonical_dir),
                    "--work-dir",
                    str(root / "work"),
                    "--output-dir",
                    str(out_dir),
                    "--ratios",
                    "0,50,100",
                    "--bag-size",
                    "4",
                    "--bags-per-stratum",
                    "1",
                    "--seed",
                    "11",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            bag_keys = np.load(out_dir / "bag_keys.npy")
            bag_targets = np.load(out_dir / "bag_targets.npy")
            self.assertEqual(bag_keys.shape[0], 12)
            self.assertEqual(sorted(np.unique(bag_targets).tolist()), [0.0, 0.5, 1.0])
            self.assertEqual([int((bag_keys == key).sum()) for key in sorted(np.unique(bag_keys))], [4, 4, 4])

    def test_promote_control_eval_helpers(self):
        from validate.evaluate_promote_control import (
            DatasetSpec,
            build_dataset_specs,
            monotonicity_check,
            parse_mix_dataset,
        )

        parsed = parse_mix_dataset("25:/tmp/mix25")
        self.assertEqual(parsed.name, "mix_25")
        self.assertEqual(parsed.ratio, 25.0)

        class Args:
            ivt_dir = "/tmp/ivt"
            full_mod_dir = "/tmp/full"
            mix_dataset = ["50:/tmp/mix50"]

        specs = build_dataset_specs(Args())
        self.assertEqual(
            [(item.name, item.ratio) for item in specs],
            [("ivt", 0.0), ("mix_50", 50.0), ("full_mod", 100.0)],
        )

        monotonic = monotonicity_check([
            {"name": "ivt", "ratio": 0.0, "mean_pred_mod_prob": 0.1},
            {"name": "mix_50", "ratio": 50.0, "mean_pred_mod_prob": 0.5},
            {"name": "full_mod", "ratio": 100.0, "mean_pred_mod_prob": 0.9},
        ])
        self.assertTrue(monotonic["non_decreasing_by_mean_prob"])

    def test_basecaller_defaults_to_koi(self):
        from tetramod.cli.basecaller import argparser

        args = argparser().parse_args(["model_dir", "reads_dir"])
        self.assertTrue(args.use_koi)

        args = argparser().parse_args(["model_dir", "reads_dir", "--no-use-koi"])
        self.assertFalse(args.use_koi)


class MultiHeadKoiSmokeTest(unittest.TestCase):
    @staticmethod
    def _tiny_config():
        return {
            "model": {
                "package": "tetramod.transformer.multihead_model",
                "d_model": 8,
                "nhead": 2,
                "dim_feedforward": 16,
                "num_layers": 0,
                "kernel_size": 3,
                "stride": 2,
                "mod_bases": ["A", "C"],
                "mod_global_labels": ["canonical_A", "canonical_C", "m6A"],
                "mod_head_defs": {
                    "A": ["canonical_A", "m6A"],
                    "C": ["canonical_C"],
                },
                "mod_trunk_dim": 4,
                "mod_trunk_kernel_size": 3,
                "mod_trunk_depth": 0,
                "mod_head_dropout": 0.0,
                "blank_score": 2.0,
                "expand_blanks": True,
            },
            "input": {
                "features": 1,
                "n_pre_post_context_bases": [0, 0],
            },
            "labels": {
                "labels": ["N", "A", "C"],
            },
            "global_norm": {
                "state_len": 2,
            },
            "training": {},
        }

    def test_use_koi_switches_crf_to_raw_scores_and_preserves_expand_helper(self):
        import torch

        from tetramod.transformer.multihead_model import MultiHeadModel

        model = MultiHeadModel(self._tiny_config())
        model.eval()

        with torch.inference_mode():
            expanded = model(torch.randn(1, 1, 16))["base_scores"]

        self.assertEqual(expanded.shape[-1], model.seqdist.n_score())
        self.assertTrue(model.crf.expand_blanks)

        model.use_koi()
        self.assertFalse(model.crf.expand_blanks)

        with torch.inference_mode():
            raw = model(torch.randn(1, 1, 16))["base_scores"]

        self.assertEqual(raw.shape[-1], model.seqdist.n_base ** (model.seqdist.state_len + 1))
        self.assertEqual(model.expand_base_scores(raw).shape[-1], model.seqdist.n_score())


if __name__ == "__main__":
    unittest.main()
