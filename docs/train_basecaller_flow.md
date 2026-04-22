# TetraMod Train/Basecaller Flow

This document covers only the migrated `bonito-mixed` commands exposed by
TetraMod:

- `tetramod train`, migrated from `bonito train_mod`
- `tetramod basecaller`, migrated from `bonito basecaller_mod`

`../bonitov1.1.0` is the upstream Bonito reference. TetraMod imports upstream
Bonito directly where possible and keeps only the modified-base code that is
absent from upstream Bonito.

## Comparison Summary

The following pieces exist in `../bonito-mixed` but not in upstream
`../bonitov1.1.0`, so TetraMod carries them as native project modules:

- `bonito/cli/train_mod.py` -> `tetramod.cli.train`
- `bonito/cli/basecaller_mod.py` -> `tetramod.cli.basecaller`
- `bonito/training_mod.py` -> `tetramod.training_mod`
- `bonito/train_mod_data.py` -> `tetramod.train_mod_data`
- `bonito/transformer/multihead_model.py` -> `tetramod.transformer.multihead_model`
- `bonito/transformer/multihead_basecall.py` -> `tetramod.transformer.multihead_basecall`
- `bonito/models/configs/multihead_transformer.toml` -> packaged under
  `tetramod.models.configs`

The following runtime pieces are upstream Bonito and are imported directly from
the installed `bonito` package:

- model/data settings: `bonito.data`
- IO and SAM/FASTQ writing: `bonito.io`
- POD5/raw read loading: `bonito.reader`, `bonito.pod5`
- alignment: `bonito.aligner`
- neural network layers and CRF primitives: `bonito.nn`, `bonito.crf`
- multiprocessing and scheduling helpers: `bonito.multiprocessing`,
  `bonito.schedule`
- common utility helpers: selected functions from `bonito.util`

## `tetramod train`

Entry point: `tetramod.cli.train`.

Flow:

1. Parse arguments matching `bonito-mixed/bonito/cli/train_mod.py`.
2. Load the TetraMod packaged multi-head config unless `--config` is provided.
3. Load the pretrained basecaller config through the Bonito-compatible model
   directory resolver.
4. Copy runtime fields from the pretrained basecaller config into the train-mod
   config:
   `basecaller`, `scaling`, `standardisation`, `normalisation`, `run_info`,
   `qscore`, labels, input features, and state length.
5. Validate that labels, input features, state length, and signal
   normalization match the pretrained basecaller.
6. Build training metadata with mode `standalone_mod_head` and record the
   pretrained basecaller path.
7. Construct `tetramod.transformer.multihead_model.Model`.
   This model reuses upstream Bonito CRF and NN layers but adds modification
   heads and standalone basecaller-encoder handling.
8. Load matching pretrained basecaller weights into the frozen basecaller
   encoder using TetraMod's modified-base weight-loading helper.
9. Load modified-base training data via
   `tetramod.train_mod_data.load_train_mod_data`.
   The dataset expects Bonito-style arrays plus `mod_targets.npy`.
10. Write the resolved run `config.toml` into the training directory.
11. Train with `tetramod.training_mod.TrainerMod`.
    This trainer reuses upstream Bonito scheduler, CSV logging, accuracy, and
    utility functions, but its batch/loss path handles `(base, mod)` targets.
12. Save standalone mod-head checkpoints as `weights_{epoch}.tar`.

## `tetramod basecaller`

Entry point: `tetramod.cli.basecaller`.

Flow:

1. Parse arguments matching `bonito-mixed/bonito/cli/basecaller_mod.py`.
2. Reject unsupported `--use-koi` and `--revcomp` paths from the
   `bonito-mixed` minimal modified-base basecaller.
3. Read input files with upstream `bonito.reader.Reader`.
4. Initialize the runtime through upstream Bonito seeding/device utilities.
5. Select FASTQ/SAM/BAM/CRAM output mode with upstream `bonito.io.biofmt`.
6. Resolve/download model names using upstream `bonito.cli.download` helpers.
7. Load the model with TetraMod's Bonito-compatible model loader:
   - it reuses upstream Bonito config defaults and checkpoint discovery;
   - it aliases old `bonito.transformer.multihead_model` configs to
     `tetramod.transformer.multihead_model`;
   - it reconstructs standalone mod-head models by first loading the recorded
     pretrained basecaller.
8. Fuse batchnorm through upstream `bonito.nn.fuse_bn_`.
9. Require the loaded model to provide `predict_mods()`.
10. Optionally create an upstream `bonito.aligner.Aligner`.
11. Read signals through upstream Bonito reader APIs, using the model's recorded
    scaling and normalization configuration.
12. Run multi-head basecalling through
    `tetramod.transformer.multihead_basecall.basecall`.
    This path chunks reads with upstream Bonito utility behavior, decodes CRF
    basecalls, calls `model.predict_mods()`, and emits MM/ML modified-base tags.
13. Optionally align records with upstream `bonito.aligner.align_map`.
14. Write output records through upstream `bonito.io.Writer`.

## Remaining Bonito Dependency

TetraMod does not vendor upstream Bonito. A runtime installation that provides
`import bonito` is still required. TetraMod only carries the modified-base
extensions that upstream Bonito v1.1.0 does not provide.
