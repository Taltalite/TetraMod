import unittest
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_repo_script(relative_path):
    path = Path(relative_path)
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


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
        self.assertFalse(args.compile_model)

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
                "--llp-loss",
                "huber",
                "--llp-tolerance",
                "0.05",
                "--llp-huber-delta",
                "0.1",
                "--init-promote-checkpoint",
                "stage1_model",
                "--compile",
            ]
        )
        self.assertEqual(args.promote_stage, "llp")
        self.assertEqual(args.llp_proportion, 25.0)
        self.assertEqual(args.llp_bag_size, 8)
        self.assertEqual(args.llp_loss, "huber")
        self.assertEqual(args.llp_tolerance, 0.05)
        self.assertEqual(args.llp_huber_delta, 0.1)
        self.assertEqual(args.init_promote_checkpoint, Path("stage1_model"))
        self.assertTrue(args.compile_model)

    def test_train_promote_init_checkpoint_resolver(self):
        from tetramod.cli.train_promote import (
            has_training_checkpoints,
            resolve_init_promote_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            weights_1 = model_dir / "weights_1.tar"
            weights_3 = model_dir / "weights_3.tar"
            weights_1.write_bytes(b"stage1")
            weights_3.write_bytes(b"stage3")
            (model_dir / "weights_bad.tar").write_bytes(b"ignored")

            self.assertEqual(resolve_init_promote_checkpoint(model_dir), weights_3.resolve())
            self.assertEqual(resolve_init_promote_checkpoint(weights_1), weights_1.resolve())
            self.assertTrue(has_training_checkpoints(model_dir))

            empty_dir = model_dir / "empty"
            empty_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                resolve_init_promote_checkpoint(empty_dir)
            with self.assertRaises(FileNotFoundError):
                resolve_init_promote_checkpoint(model_dir / "missing")

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
            DEFAULT_LLP_HUBER_DELTA,
            DEFAULT_LLP_LOSS,
            DEFAULT_LLP_TOLERANCE,
            normalize_llp_proportion,
            resolve_llp_settings,
            resolve_promote_stage,
        )

        config = {"training": {"promote_stage": "control"}}

        self.assertEqual(resolve_promote_stage(config), PROMOTE_STAGE_CONTROL)
        self.assertEqual(resolve_promote_stage({}, "control"), PROMOTE_STAGE_CONTROL)
        self.assertEqual(resolve_promote_stage({}, "llp"), PROMOTE_STAGE_LLP)
        self.assertEqual(CONTROL_WARMUP_LOSS_PATH, "a_head_control_warmup_viterbi_bce")
        self.assertEqual(LLP_LOSS_PATH, "a_head_llp_mean_pool_proportion")
        self.assertEqual(normalize_llp_proportion(25), 0.25)
        self.assertEqual(
            resolve_llp_settings({"training": {"llp_proportion": 0.5, "llp_bag_size": 4}}),
            {
                "llp_proportion": 0.5,
                "llp_bag_size": 4,
                "llp_loss": DEFAULT_LLP_LOSS,
                "llp_tolerance": DEFAULT_LLP_TOLERANCE,
                "llp_huber_delta": DEFAULT_LLP_HUBER_DELTA,
            },
        )
        self.assertEqual(
            resolve_llp_settings(
                {"training": {"llp_bag_size": 4}},
                cli_loss="mse",
                cli_tolerance=0.05,
                cli_huber_delta=0.1,
            ),
            {
                "llp_bag_size": 4,
                "llp_loss": "mse",
                "llp_tolerance": 0.05,
                "llp_huber_delta": 0.1,
            },
        )

    def test_promote_can_extract_legacy_rna002_encoder_config(self):
        from tetramod.cli.train import (
            extract_pretrained_encoder_config,
            merge_pretrained_runtime_config,
            validate_pretrained_runtime_config,
        )

        pretrained_config = {
            "model": {"package": "bonito.crf"},
            "labels": {"labels": ["N", "A", "C", "G", "T"]},
            "input": {"features": 1},
            "global_norm": {"state_len": 5},
            "encoder": {
                "activation": "swish",
                "stride": 5,
                "rnn_type": "lstm",
                "features": 768,
                "scale": 5.0,
                "winlen": 19,
                "blank_score": 2.0,
            },
            "normalisation": {"quantile_a": 0.2, "quantile_b": 0.8},
        }
        config = {"model": {}, "labels": {}, "input": {}, "global_norm": {}}

        encoder = extract_pretrained_encoder_config(pretrained_config)
        self.assertEqual(encoder["rnn_type"], "lstm")

        merged = merge_pretrained_runtime_config(config, pretrained_config)
        validate_pretrained_runtime_config(merged, pretrained_config)
        self.assertEqual(merged["labels"]["labels"], ["N", "A", "C", "G", "T"])
        self.assertEqual(merged["input"]["features"], 1)
        self.assertEqual(merged["global_norm"]["state_len"], 5)
        self.assertIn("normalisation", merged)

    def test_multihead_model_reconstructs_legacy_rna002_encoder(self):
        import torch

        from tetramod.transformer.multihead_model import MultiHeadModel

        config = {
            "model": {
                "package": "tetramod.transformer.multihead_model",
                "pretrained_encoder": {
                    "activation": "swish",
                    "stride": 5,
                    "rnn_type": "lstm",
                    "features": 8,
                    "scale": 5.0,
                    "winlen": 5,
                    "blank_score": 2.0,
                    "num_layers": 0,
                },
                "mod_bases": ["A"],
                "mod_global_labels": ["canonical_A", "m6A"],
                "mod_head_defs": {"A": ["canonical_A", "m6A"]},
                "mod_trunk_dim": 4,
                "mod_trunk_depth": 0,
                "mod_head_dropout": 0.0,
            },
            "input": {"features": 1},
            "labels": {"labels": ["N", "A", "C"]},
            "global_norm": {"state_len": 2},
            "training": {"mode": "standalone_mod_head"},
        }

        model = MultiHeadModel(config)
        outputs = model(torch.zeros((2, 1, 60), dtype=torch.float32))

        self.assertEqual(model.stride, 5)
        self.assertEqual(outputs["base_scores"].shape[1], 2)
        self.assertEqual(outputs["mod_logits_by_base"]["A"].shape[:2], (2, outputs["base_scores"].shape[0]))
        self.assertFalse(any(param.requires_grad for param in model.encoder.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.mod_heads.parameters()))

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
        expected = -(torch.tensor(0.75) * bag_prob.clamp(1e-6, 1.0 - 1e-6).log())
        expected = expected - (torch.tensor(0.25) * (1.0 - bag_prob.clamp(1e-6, 1.0 - 1e-6)).log())
        self.assertTrue(torch.allclose(losses["llp_loss"], expected))
        self.assertTrue(torch.allclose(losses["total_loss"], expected * 2.0))
        self.assertEqual(losses["llp_num_bags"].item(), 1.0)
        self.assertEqual(losses["llp_num_reads"].item(), 2.0)
        self.assertIn("llp_bag_mae", losses)
        self.assertIn("llp_bag_rmse", losses)
        self.assertIn("llp_bag_bias", losses)
        self.assertIn("llp_bag_corr", losses)

    def test_promote_llp_relaxed_loss_zeros_inside_tolerance(self):
        import torch

        from tetramod.training_promote import LLPProportionLoss

        class FakeModel:
            mod_bases = ["A"]
            mod_head_defs = {"A": ["canonical_A", "m6A"]}
            standalone_mod_head = True
            mod_loss_weight = 1.0

            def __init__(self, logits):
                self.logits = logits

            def align_predictions_to_targets(self, outputs, targets, target_lengths, mod_targets):
                return {
                    "per_head": {
                        "A": {
                            "flat_logits": self.logits,
                            "flat_targets": torch.zeros((2,), dtype=torch.long),
                            "flat_sample_indices": torch.tensor([0, 1]),
                        }
                    }
                }

        logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
        criterion = LLPProportionLoss(
            FakeModel(logits),
            llp_proportion=0.55,
            llp_loss="mse",
            llp_tolerance=0.1,
        )
        outputs = {"base_scores": torch.zeros((1, 2, 1)), "bag_keys": torch.tensor([7, 7])}

        losses = criterion(
            outputs,
            targets=torch.zeros((2, 1), dtype=torch.long),
            target_lengths=torch.ones((2,), dtype=torch.long),
            mod_targets=torch.zeros((2, 1), dtype=torch.long),
        )

        self.assertTrue(torch.allclose(losses["llp_loss"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(losses["llp_bag_mae"], torch.tensor(0.05)))
        self.assertTrue(torch.allclose(losses["llp_mean_bag_prob"], torch.tensor(0.5)))
        self.assertTrue(torch.allclose(losses["llp_mean_bag_target"], torch.tensor(0.55)))

    def test_promote_llp_huber_uses_relaxed_boundary_outside_tolerance(self):
        import torch
        import torch.nn.functional as F

        from tetramod.training_promote import LLPProportionLoss

        class FakeModel:
            mod_bases = ["A"]
            mod_head_defs = {"A": ["canonical_A", "m6A"]}
            standalone_mod_head = True
            mod_loss_weight = 1.0

            def __init__(self, logits):
                self.logits = logits

            def align_predictions_to_targets(self, outputs, targets, target_lengths, mod_targets):
                return {
                    "per_head": {
                        "A": {
                            "flat_logits": self.logits,
                            "flat_targets": torch.zeros((2,), dtype=torch.long),
                            "flat_sample_indices": torch.tensor([0, 1]),
                        }
                    }
                }

        logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]], requires_grad=True)
        criterion = LLPProportionLoss(
            FakeModel(logits),
            llp_proportion=0.4,
            llp_loss="huber",
            llp_tolerance=0.05,
            llp_huber_delta=0.1,
        )
        outputs = {"base_scores": torch.zeros((1, 2, 1)), "bag_keys": torch.tensor([3, 3])}

        losses = criterion(
            outputs,
            targets=torch.zeros((2, 1), dtype=torch.long),
            target_lengths=torch.ones((2,), dtype=torch.long),
            mod_targets=torch.zeros((2, 1), dtype=torch.long),
        )

        bag_prob = torch.sigmoid(torch.tensor(2.0))
        expected = F.huber_loss(bag_prob, torch.tensor(0.45), delta=0.1)
        self.assertTrue(torch.allclose(losses["llp_loss"], expected))

    def test_promote_llp_loss_is_safe_under_autocast(self):
        import torch

        from tetramod.training_promote import LLPProportionLoss

        class FakeModel:
            mod_bases = ["A"]
            mod_head_defs = {"A": ["canonical_A", "m6A"]}
            standalone_mod_head = True
            mod_loss_weight = 1.0

            def __init__(self, logits):
                self.logits = logits

            def align_predictions_to_targets(self, outputs, targets, target_lengths, mod_targets):
                return {
                    "per_head": {
                        "A": {
                            "flat_logits": self.logits,
                            "flat_targets": torch.zeros((2,), dtype=torch.long),
                            "flat_sample_indices": torch.tensor([0, 1]),
                        }
                    }
                }

        logits = torch.tensor([[0.0, 1.0], [0.0, 2.0]], requires_grad=True)
        criterion = LLPProportionLoss(FakeModel(logits), llp_proportion=0.5)
        outputs = {"base_scores": torch.zeros((1, 2, 1)), "bag_keys": torch.tensor([1, 1])}
        with torch.autocast("cpu", enabled=True):
            losses = criterion(
                outputs,
                targets=torch.zeros((2, 1), dtype=torch.long),
                target_lengths=torch.ones((2,), dtype=torch.long),
                mod_targets=torch.zeros((2, 1), dtype=torch.long),
            )
        losses["total_loss"].backward()
        self.assertTrue(torch.isfinite(losses["llp_loss"]))
        self.assertIsNotNone(logits.grad)

    def test_synthetic_llp_controls_builder_smoke(self):
        import subprocess
        import sys
        import tempfile

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

    def test_llp_candidate_mod_targets_mark_a_sites_without_positive_labels(self):
        import numpy as np

        make_targets = load_repo_script("gen_data/make_mod_targets_m6a.py")
        references = np.asarray([[1, 2, 1, 4, 0], [3, 1, 4, 0, 0]], dtype=np.uint8)
        lengths = np.asarray([4, 3], dtype=np.uint16)

        mod_targets = make_targets.build_mod_targets(
            references,
            lengths,
            make_targets.MODE_LLP_CANDIDATE,
            make_targets.NON_A_POLICY_IGNORE,
            -100,
        )

        self.assertEqual(mod_targets[0, 0], make_targets.CANONICAL_LABELS[make_targets.BASE_A])
        self.assertEqual(mod_targets[0, 2], make_targets.CANONICAL_LABELS[make_targets.BASE_A])
        self.assertEqual(mod_targets[1, 1], make_targets.CANONICAL_LABELS[make_targets.BASE_A])
        self.assertFalse(bool((mod_targets == make_targets.M6A_LABEL).any()))
        self.assertEqual(mod_targets[0, 1], -100)
        self.assertEqual(mod_targets[0, 4], -100)

    def test_mafia_synthetic_manifest_parses_center_m6a(self):
        mafia = load_repo_script("gen_data/create_mafia_synthetic_stage1_dataset.py")

        sequence, center = mafia.parse_modified_sequence("GGACU/m6AGGACC")

        self.assertEqual(sequence, "GGACTAGGACC")
        self.assertEqual(center, 5)
        self.assertEqual(sequence[center], "A")

        plain_sequence, plain_center = mafia.parse_modified_sequence("CCGGACTAA", center_index=4)
        self.assertEqual(plain_sequence, "CCGGACTAA")
        self.assertEqual(plain_center, 4)

    def test_mafia_synthetic_matches_only_controlled_centers(self):
        import numpy as np

        mafia = load_repo_script("gen_data/create_mafia_synthetic_stage1_dataset.py")
        oligo = mafia.OligoSpec(
            oligo_id="RL_M0_S0",
            sequence="GGACT",
            center_index=2,
            motif="GGACT",
            ligation_strategy="random_ligation",
        )
        run = mafia.RunSpec(
            run_id="run_mod",
            accession="ERR0",
            local_name="run_mod",
            modification_status="modified",
            ligation_strategy="random_ligation",
            oligo_ids=("RL_M0_S0",),
        )

        units = mafia.find_oligo_units(
            "TTTGGACTAAA",
            [oligo],
            run,
            min_identity=1.0,
            max_mismatches=0,
            allow_reverse_match=False,
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].center_index, 5)
        self.assertEqual(units[0].label, mafia.M6A_LABEL)

        emitted = np.arange(0, 55, 5, dtype=np.int64)
        assigned = mafia.assign_units_to_windows(units, emitted, [(0, 30), (25, 55)])
        self.assertEqual(len(assigned), 1)
        self.assertEqual(next(iter(assigned.values()))[0].oligo.oligo_id, "RL_M0_S0")

    def test_mafia_synthetic_run_manifest_supports_mixed_labels(self):
        mafia = load_repo_script("gen_data/create_mafia_synthetic_stage1_dataset.py")
        mixed = mafia.RunSpec(
            run_id="test1",
            accession="ERR1",
            local_name="test1",
            modification_status="mixed",
            ligation_strategy="random_ligation",
            oligo_ids=("RL_M4_S0", "RL_M5_S0"),
            modified_oligo_ids=("RL_M4_S0",),
        )

        self.assertEqual(mafia.label_for_match(mixed, "RL_M4_S0"), mafia.M6A_LABEL)
        self.assertEqual(mafia.label_for_match(mixed, "RL_M5_S0"), mafia.CANONICAL_A_LABEL)

    def test_mafia_stage1_merger_balances_train_split(self):
        import subprocess
        import sys
        import tempfile

        import numpy as np

        def write_dataset(directory, label_value, status):
            directory.mkdir(parents=True)
            n = 8
            np.save(directory / "chunks.npy", np.ones((n, 8), dtype=np.float16))
            np.save(directory / "references.npy", np.asarray([[1, 2, 1]] * n, dtype=np.uint8))
            np.save(directory / "reference_lengths.npy", np.full((n,), 3, dtype=np.uint16))
            mod_targets = np.full((n, 3), -100, dtype=np.int16)
            mod_targets[:, 1] = label_value
            np.save(directory / "mod_targets.npy", mod_targets)
            np.savez(
                directory / "metadata.npz",
                record_id=np.asarray([f"{status}_{idx}" for idx in range(n)], dtype=str),
                pod5_read_id=np.asarray([f"pod5_{status}_{idx}" for idx in range(n)], dtype=str),
                run_id=np.asarray([f"run_{status}"] * n, dtype=str),
                contig=np.asarray(["mafia_synthetic"] * n, dtype=str),
                primary_site_key=np.asarray([f"oligo:{idx}:1:A" for idx in range(n)], dtype=str),
                kmer_context=np.asarray(["GGACT"] * n, dtype=str),
                motif_context=np.asarray(["GGACT"] * n, dtype=str),
                oligo_ids=np.asarray(["oligo"] * n, dtype=str),
                oligo_motifs=np.asarray(["GGACT"] * n, dtype=str),
                oligo_orientations=np.asarray(["+"] * n, dtype=str),
                modification_status=np.asarray([status] * n, dtype=str),
                ligation_strategy=np.asarray(["random_ligation"] * n, dtype=str),
                ref_start=np.zeros((n,), dtype=np.int64),
                ref_end=np.full((n,), 3, dtype=np.int64),
                ref_strand=np.ones((n,), dtype=np.int8),
                chunk_start=np.zeros((n,), dtype=np.int64),
                chunk_end=np.full((n,), 8, dtype=np.int64),
                primary_site_pos=np.ones((n,), dtype=np.int64),
                mean_qscore=np.full((n,), 12.0, dtype=np.float32),
                mapping_accuracy=np.ones((n,), dtype=np.float32),
                mapping_coverage=np.ones((n,), dtype=np.float32),
                labeled_center_count=np.ones((n,), dtype=np.int16),
                positive_center_count=np.full((n,), int(label_value == 4), dtype=np.int16),
                negative_center_count=np.full((n,), int(label_value == 0), dtype=np.int16),
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pos = root / "pos"
            neg = root / "neg"
            out = root / "merged"
            write_dataset(pos, 4, "modified")
            write_dataset(neg, 0, "unmodified")

            subprocess.run(
                [
                    sys.executable,
                    "gen_data/merge_mafia_stage1_datasets.py",
                    "--dataset",
                    f"pos:{pos}",
                    "--dataset",
                    f"neg:{neg}",
                    "--output-dir",
                    str(out),
                    "--valid-fraction",
                    "0.25",
                    "--seed",
                    "3",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            train_mods = np.load(out / "mod_targets.npy")
            valid_mods = np.load(out / "validation" / "mod_targets.npy")
            self.assertEqual(int((train_mods == 4).sum()), int((train_mods == 0).sum()))
            self.assertGreater(int((valid_mods == 4).sum()), 0)
            self.assertGreater(int((valid_mods == 0).sum()), 0)

    def test_real_llp_ratio_ivt_builder_parses_fractional_runs(self):
        builder = load_repo_script("gen_data/build_real_llp_from_ratio_ivt.py")

        ratio_run = builder.parse_ratio_run("12.5:/tmp/ratio12.bam:/tmp/pod5:rna002_12p5")
        self.assertEqual(ratio_run.ratio_label, "12.5")
        self.assertAlmostEqual(ratio_run.ratio_fraction, 0.125)
        self.assertEqual(str(ratio_run.bam), "/tmp/ratio12.bam")
        self.assertEqual(str(ratio_run.pod5_dir), "/tmp/pod5")
        self.assertEqual(ratio_run.run_id, "rna002_12p5")

        ratio_dataset = builder.parse_ratio_dataset("75:/tmp/dataset75")
        self.assertEqual(ratio_dataset.ratio_label, "75")
        self.assertAlmostEqual(ratio_dataset.ratio_fraction, 0.75)

    def test_real_llp_ratio_stratified_builder_does_not_require_common_strata(self):
        import subprocess
        import sys
        import tempfile

        import numpy as np

        def write_ratio_dataset(directory, ratio_name, run_id, site_prefix):
            directory.mkdir(parents=True)
            n = 8
            np.save(directory / "chunks.npy", np.ones((n, 8), dtype=np.float16))
            np.save(directory / "references.npy", np.ones((n, 3), dtype=np.uint8))
            np.save(directory / "reference_lengths.npy", np.full((n,), 3, dtype=np.uint16))
            np.save(directory / "mod_targets.npy", np.zeros((n, 3), dtype=np.int16))
            np.savez(
                directory / "metadata.npz",
                record_id=np.asarray([f"{ratio_name}_read_{idx}" for idx in range(n)], dtype=str),
                pod5_read_id=np.asarray([f"{ratio_name}_pod5_{idx}" for idx in range(n)], dtype=str),
                run_id=np.asarray([run_id] * n, dtype=str),
                contig=np.asarray(["tx1"] * n, dtype=str),
                primary_site_key=np.asarray([f"{site_prefix}:{100 + idx}:1:A" for idx in range(n)], dtype=str),
                kmer_context=np.asarray(["GGACT"] * n, dtype=str),
                motif_context=np.asarray(["DRACH"] * n, dtype=str),
                ref_start=np.arange(n, dtype=np.int64),
                ref_end=np.arange(n, dtype=np.int64) + 3,
                ref_strand=np.ones((n,), dtype=np.int8),
                chunk_start=np.zeros((n,), dtype=np.int64),
                chunk_end=np.full((n,), 8, dtype=np.int64),
                primary_site_pos=np.arange(n, dtype=np.int64) + 100,
                mean_qscore=np.full((n,), 12.0, dtype=np.float32),
                mapping_accuracy=np.full((n,), 0.99, dtype=np.float32),
                mapping_coverage=np.full((n,), 0.95, dtype=np.float32),
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            d12 = root / "ratio_12p5"
            d75 = root / "ratio_75"
            out = root / "out"
            write_ratio_dataset(d12, "12p5", "run_a", "site_a")
            write_ratio_dataset(d75, "75", "run_b", "site_b")

            subprocess.run(
                [
                    sys.executable,
                    "gen_data/build_llp_mixture_dataset.py",
                    "--ratio-dataset",
                    f"12.5:{d12}",
                    "--ratio-dataset",
                    f"75:{d75}",
                    "--output-dir",
                    str(out),
                    "--bagging-mode",
                    "ratio-stratified",
                    "--match-fields",
                    "contig,kmer_context,motif_context",
                    "--bag-size",
                    "4",
                    "--min-bag-size",
                    "4",
                    "--seed",
                    "7",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            bag_targets = np.load(out / "bag_targets.npy")
            bag_keys = np.load(out / "bag_keys.npy")
            self.assertEqual(sorted(np.unique(bag_targets).tolist()), [0.125, 0.75])
            self.assertEqual(sorted([int((bag_keys == key).sum()) for key in np.unique(bag_keys)]), [4, 4, 4, 4])

    def test_llp_dataset_diagnostics_script_smoke(self):
        import json
        import subprocess
        import sys
        import tempfile

        import numpy as np

        def write_split(directory):
            directory.mkdir(parents=True)
            n = 8
            np.save(directory / "chunks.npy", np.ones((n, 8), dtype=np.float16))
            np.save(directory / "references.npy", np.ones((n, 3), dtype=np.uint8))
            np.save(directory / "reference_lengths.npy", np.full((n,), 3, dtype=np.uint16))
            mod_targets = np.full((n, 3), -100, dtype=np.int16)
            mod_targets[:, :2] = 0
            np.save(directory / "mod_targets.npy", mod_targets)
            np.save(directory / "bag_keys.npy", np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64))
            np.save(directory / "bag_targets.npy", np.asarray([0.5] * 4 + [0.75] * 4, dtype=np.float32))
            np.save(directory / "ratio_labels.npy", np.asarray(["50"] * 4 + ["75"] * 4, dtype=str))
            np.savez(
                directory / "metadata.npz",
                record_id=np.asarray([f"read_{idx}" for idx in range(n)], dtype=str),
                pod5_read_id=np.asarray([f"pod5_{idx}" for idx in range(n)], dtype=str),
                run_id=np.asarray(["run_a"] * 4 + ["run_b"] * 4, dtype=str),
                contig=np.asarray(["tx1"] * n, dtype=str),
                primary_site_key=np.asarray(["tx1:100:1:A"] * 4 + ["tx1:200:1:A"] * 4, dtype=str),
                kmer_context=np.asarray(["GGACT"] * 4 + ["TTACA"] * 4, dtype=str),
                motif_context=np.asarray(["DRACH"] * 4 + ["UUACH"] * 4, dtype=str),
                mean_qscore=np.asarray([12.0] * 4 + [10.0] * 4, dtype=np.float32),
                mapping_coverage=np.asarray([0.95] * 4 + [0.85] * 4, dtype=np.float32),
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            output = root / "diagnostics"
            write_split(dataset)
            write_split(dataset / "validation")

            subprocess.run(
                [
                    sys.executable,
                    "validate/diagnose_llp_dataset.py",
                    str(dataset),
                    "--output-dir",
                    str(output),
                    "--split",
                    "all",
                    "--compare-ratios",
                    "50,75",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["splits"]["train"]["num_reads"], 8)
            self.assertEqual(summary["splits"]["valid"]["num_bags"], 2)
            self.assertTrue((output / "ratio_summary.tsv").exists())
            self.assertTrue((output / "category_distance.tsv").exists())

    def test_create_dataset_rna002_resolves_model_config_normalisation(self):
        import tempfile

        create_dataset = load_repo_script("gen_data/create_dataset_dorado_ctc_like.py")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.toml"
            config_path.write_text(
                """
[normalisation]
quantile_a = 0.2
quantile_b = 0.8
shift_multiplier = 0.48
scale_multiplier = 0.59
""",
                encoding="utf-8",
            )

            args = create_dataset.parse_args(
                [
                    "--bam-file",
                    "/tmp/in.bam",
                    "--pod5-dir",
                    "/tmp/pod5",
                    "--reference-fasta",
                    "/tmp/ref.fa",
                    "--output-dir",
                    "/tmp/out",
                    "--rna002",
                    "--model-config",
                    str(config_path),
                ]
            )
            create_dataset.resolve_signal_normalisation(args)

        self.assertEqual(args.norm_strategy, "quantile")
        self.assertEqual(args.quantile_a, 0.2)
        self.assertEqual(args.quantile_b, 0.8)
        self.assertEqual(args.shift_multiplier, 0.48)
        self.assertEqual(args.scale_multiplier, 0.59)

        default_args = create_dataset.parse_args(
            [
                "--bam-file",
                "/tmp/in.bam",
                "--pod5-dir",
                "/tmp/pod5",
                "--reference-fasta",
                "/tmp/ref.fa",
                "--output-dir",
                "/tmp/out",
            ]
        )
        create_dataset.resolve_signal_normalisation(default_args)
        self.assertEqual(default_args.norm_strategy, "from-bam")

    def test_fast5_tar_to_pod5_converter_handles_single_read_fast5(self):
        import json
        import subprocess
        import sys
        import tarfile
        import tempfile
        import uuid

        import h5py
        import numpy as np
        import pod5

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fast5_path = root / "read.fast5"
            read_id = str(uuid.uuid4())
            with h5py.File(fast5_path, "w") as h5:
                unique = h5.create_group("UniqueGlobalKey")
                channel = unique.create_group("channel_id")
                channel.attrs["channel_number"] = "42"
                channel.attrs["digitisation"] = 8192.0
                channel.attrs["offset"] = 10.0
                channel.attrs["range"] = 1467.6
                channel.attrs["sampling_rate"] = 4000
                tracking = unique.create_group("tracking_id")
                tracking.attrs["run_id"] = "run1"
                tracking.attrs["exp_start_time"] = "2020-01-01T00:00:00Z"
                tracking.attrs["protocol_start_time"] = "2020-01-01T00:00:00Z"
                tracking.attrs["protocol_run_id"] = "proto1"
                tracking.attrs["sample_id"] = "sample1"
                context = unique.create_group("context_tags")
                context.attrs["sequencing_kit"] = "SQK-RNA002"
                read = h5.create_group("Raw").create_group("Reads").create_group("Read_1")
                read.attrs["read_id"] = read_id
                read.attrs["read_number"] = 1
                read.attrs["start_time"] = 0
                read.attrs["start_mux"] = 1
                read.attrs["median_before"] = 80.0
                read.create_dataset("Signal", data=np.arange(10, dtype=np.int16))

            inner_archive_path = root / "1204670-1.fast5.tar"
            with tarfile.open(inner_archive_path, "w") as archive:
                archive.add(fast5_path, arcname="read.fast5")
            archive_path = root / "RNAAB089716.fast5.tar.gz.4"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(inner_archive_path, arcname="fast5/1204670-1.fast5.tar")
            output_dir = root / "pod5"

            subprocess.run(
                [
                    sys.executable,
                    "gen_data/convert_fast5_tar_to_pod5.py",
                    str(archive_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            output_path = output_dir / "RNAAB089716.part4.pod5"
            with pod5.Reader(output_path) as reader:
                records = list(reader.reads())
            self.assertEqual(len(records), 1)
            self.assertEqual(str(records[0].read_id), read_id)

            summary = json.loads((output_dir / "fast5_to_pod5_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["inputs"][0]["observed_pod5_reads"], 1)
            self.assertEqual(summary["inputs"][0]["failed_files"], [])

    def test_promote_control_eval_helpers(self):
        from validate.evaluate_promote_control import (
            DatasetSpec,
            build_dataset_specs,
            clear_alignment_cache,
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

        class FakeModel:
            def __init__(self):
                self._alignment_cache = {1: "cached"}
                self.reset_called = False

            def reset_alignment_cache_stats(self):
                self.reset_called = True

        fake_model = FakeModel()
        clear_alignment_cache(fake_model)
        self.assertEqual(fake_model._alignment_cache, {})
        self.assertTrue(fake_model.reset_called)

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
