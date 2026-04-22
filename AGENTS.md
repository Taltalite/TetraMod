# AGENTS.md

## Scope

This repository is the only writable target.

Reference directories:
- ../bonitov1.1.0

Do not modify either reference directory.

## Current goal

The current task is to debug and fix behavior differences between:
- `tetramod basecaller`
- `bonito basecaller`

The target is functional parity where intended.

## Rules

- Focus on `basecaller` only unless the evidence clearly shows a shared dependency also affects `train`.
- Prefer diagnosis first, then minimal code changes.
- Reproduce the discrepancy before changing code.
- Compare against:
  - `../bonitov1.1.0` for upstream baseline behavior
- Do not do broad refactors.
- Do not rename modules or reorganize the package unless required for the fix.
- Keep changes small and reviewable.
- Add or update a minimal regression test or smoke check whenever a discrepancy is fixed.
- At the end of each task, report:
  1. reproduced difference
  2. root cause or current hypothesis
  3. files changed
  4. validation performed
  5. remaining known gaps

## Preferred workflow

1. reproduce mismatch
2. localize mismatch to argument parsing / preprocessing / model loading / decoding / writing
3. compare corresponding code paths with `../bonitov1.1.0`
4. apply minimal fix
5. validate with the same input again