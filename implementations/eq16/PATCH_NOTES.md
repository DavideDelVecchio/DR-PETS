# HalfCheetah physics correction

Updated file: `tasks.py`

The HalfCheetah adapter now calls `mujoco.mj_setConst(model, data)` after
scaling body masses and inertias, and before the subsequent `env.reset(seed=seed)`.
This is necessary to refresh constants derived from the compiled model.
The deprecated `settotalmass` in Gymnasium's official XML is left intact
on purpose; that *warning* is separate from the bug fixed here.

New regression tests: `tests/test_halfcheetah_physics.py` (three scenarios;
unit mocks do not require installing MuJoCo).

## Updating an existing EC2 run

1. Let the ongoing Python run complete, or stop it cleanly.
2. Replace the project `tasks.py` with the one in this archive.
3. Run `python -m pytest -q tests` with the active virtualenv.
4. Rerun `--task halfcheetah` mass-shift evaluations. No checkpoint
   retraining is necessary if the checkpoint was fitted on nominal data.
5. Keep the AWS instance status and spend under observation.

No automatic deployment or alteration of your current EC2 process was done.

## Live simulator check on AWS

    python cloud/verify_halfcheetah_mass.py

This checks actual compiled MuJoCo `body_subtreemass` as well as input mass
and inertia fields, so it should detect the missing `mj_setConst` call. It
was *not* executed in this packaging runtime because Gymnasium/MuJoCo is not
installed here. The mocked unit tests are independent of those packages.
