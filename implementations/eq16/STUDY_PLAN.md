# Eight independent training replicas — two AWS regions

**Objective:** exploratory/methodological PETS vs Eq. (16) DR-PETS study, not an adequately powered confirmatory superiority trial.

## Frozen stage-1 design

- **Frankfurt** `eu-central-1`: training seeds **101, 103, 107, 109**.
- **Ireland** `eu-west-1`: training seeds **113, 127, 131, 137**.
- Model: independently initialized **five-member Gaussian dynamics ensemble**, width 128, depth 3. Each replicate independently collects 10,000 nominal random transitions, trains for 25 epochs, performs one PETS online collection round (5 episodes), and retrains the new ensemble.
- Controls: CEM horizon 15, population 64, three iterations, 16 particles/model, candidate chunk 16, three independent evaluation workers (adjustable to observed hardware throughput).
- Validation: shifts mass ×0.7 / ×1.3, friction ×0.5 / ×1.5, actuator ×0.7; epsilon grid [0, 0.0001, 0.001]; **three reset seeds per shift per epsilon**, total **45**. Select the maximum **mean across shifted-condition validation rewards** (rather than maximin, aligning with the primary endpoint). Any epsilon including 0 can win.
- Held-out evaluation: nominal + five shifts; **ten paired reset seeds per condition**, PETS vs validation-selected DR-PETS; **120** episodes per training replica, length 150.
- Stage 1 totals **360 validation episodes + 960 paired-method evaluation episodes = 1,320**. The eight training replicates are the independent statistical units.
- Returns are paired on the evaluation reset seed within replicate and condition; these episode seeds are separate from the validation seeds by design. PETS and DR-PETS use the same trained ensemble within replicate.
- Primary exploratory endpoint: average across the **five shifted-condition per-replicate mean paired-return differences**; estimate one effect per independent training seed and an eight-replicate t interval. Report nominal effect, per-shift effects, and number of positive training replicates. No general robustness guarantee follows from these limited shifts.

## Stage 2: long-horizon bridge, not included in stage-1 runtime

- Reuse trained ensembles from **seed 101 in Frankfurt** and **seed 113 in Ireland** (no retraining).
- Evaluate **1,000-step** episodes on nominal, low mass, high friction, and weak motor (4 conditions), **five paired seeds** per condition per replicate, 40 evaluation episodes per replicate, **80** total.
- Use the epsilon selected using that replicate's 150-step validation. Do NOT retune epsilon on the 1,000-step bridge.
- Assess sustained mean forward velocity, displacement, control energy, reward and model fit.
- This is exploratory bridge evidence, not a full 1,000-step eight-replicate confirmation.

## Guardrails and scientific interpretation

- Model training and partial evaluation can resume only from the correct checkpoint and *exact code/physics/configuration manifest*. A training attempt interrupted before its final manifest may be retrained, not silently reused.
- Episode completions are written **as they finish** (not original submission order), using atomic CSV replacement. The in-memory collection of online trajectories is sorted back into the original seed order before retraining.
- Save full dataset and checkpoint provenance, exact physics parameters, reward decomposition, random-action baseline and held-out one-step model fit.
- The reference baseline should demonstrate positive sustained locomotion; near-zero rewards alone are not strong evidence of useful control.
- The uncertainty geometry remains **Euclidean NN parameter space**; these experiments test empirical shift robustness and do not prove a function-space Wasserstein certificate.
- Report failures and timeout codes. Never interpret more evaluation seeds from one ensemble as independent training replicates.

## Cost and elapsed time

Based on the prior L4 pilot timings, stage-1 **validation/evaluation** was estimated at around **39 L4 instance-hours TOTAL**, **not 39 hours per node**, and does not include independent training, online data collection, restarts or storage. The bridge adds roughly **12–16 L4 instance-hours**; this is extrapolated from 150-step episodes, not measured at 1,000 steps. Additional variance diagnostics and overhead remain uncertain. A planning allowance of **75–100 total GPU instance-hours** across both regions is reasonable *only as an initial authorization ceiling with monitoring*, not a guaranteed completion budget. If Ireland is T4 or other GPU, measure its relative throughput first; a 4+4 seed split may not balance wall time.

At reference On-Demand prices of about **$1.22/h Frankfurt g6.2xlarge L4** and **$0.838/h Ireland g4dn.2xlarge T4**, actual cost equals each region's instance-hours times its actual rate, plus gp3/S3 and other fees. These are price snapshots, not quotes. Starting two instances simultaneously halves elapsed time only if throughput is comparable; it does not halve aggregate GPU-hours.

## Staging recommendation

1. **Preflight** Ireland GPU vCPU quota and the chosen instance offering; the earlier account report indicated zero quota. A region with an offering but zero quota cannot launch the proposed GPU instance. Also verify available EC2 capacity, S3 access and appropriate role permissions.
2. Begin **development only**, with checkpointing every 120 seconds and instance runtime limits. Run a short 15–30 minute smoke/profiling check on each GPU to estimate cost per decision. Then launch both full 4-seed shards.
3. Download and analyze eight-replicate short-horizon results. Decide whether locomotion and estimator behavior warrant the long bridge. Run `STAGE=bridge` separately on the same result prefixes so the trained checkpoints are reused.
4. An optional future confirmatory study must have an independent, fresh training sample and prespecified powered sample size.
