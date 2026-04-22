# AGENTS.md

## Scope

This repository is the only writable target.
Reference directories:
- ../bonito-mixed
- ../bonitov1.1.0

Do not modify either reference directory.

## Goal

Migrate only these two commands from ../bonito-mixed into this repository:
- train_mod
- basecaller_mod

Expose them in this repository as:
- tetramod train
- tetramod basecaller

## Rules

- Do not migrate any other Bonito commands.
- Do not create big migration plans or architecture documents.
- Implement directly.
- If helper code is required, copy the minimum necessary code only.
- If provenance is unclear, compare against ../bonitov1.1.0 to distinguish upstream code from custom code.
- Put Bonito-derived helper code under src/tetramod/adapters/bonito/ when practical.
- Prefer small, runnable changes.
- Update pyproject.toml so the CLI can run.
- Add at least one minimal smoke test or import test.
- At the end, report:
  1. files changed
  2. what was copied from bonito-mixed
  3. any remaining dependency on Bonito