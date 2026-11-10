# Equation (16) DR-PETS: staged eight-replica two-region AWS study

**Start with [README_SPLIT_AWS.md](README_SPLIT_AWS.md)** for the verified two-region workflow. The prior pilot and full-benchmark launcher scripts have been moved to `legacy/` to prevent accidental use of a different experimental configuration.

This is a standalone research implementation; it does not copy from any branch of the user's GitHub repository. See `STUDY_PLAN.md` for the frozen experimental design and statistical limitations. The numerical Eq. (16) parameter-space score remains an empirical first-order surrogate, not a proven general function-space Wasserstein certificate.

Run `python -m pytest -q tests` before launching.
