import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO


class TetramodCliSmokeTest(unittest.TestCase):
    def test_cli_builds_expected_commands(self):
        from tetramod.cli import build_parser

        parser = build_parser()
        actions = [action for action in parser._actions if action.dest == "command"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(set(actions[0].choices), {"train", "basecaller"})

    def test_subcommand_help_parses_without_bonito_runtime_imports(self):
        from tetramod.cli import main

        for command in ("train", "basecaller"):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    main([command, "--help"])
            self.assertEqual(ctx.exception.code, 0)

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
