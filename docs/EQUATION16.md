# Numerical Equation (16) and Equation (21)

Our code evaluates a first-order distributionally robust *surrogate* for `p=2`:

\[
\widehat{J}_{\mathrm{DR}}(a)=\frac1B\sum_{b=1}^B\widehat{J}_b(a)-\epsilon\sqrt{\frac1B\sum_{b=1}^B\|\widehat g_b(a)\|_2^2}.
\]

For ensemble member `b`, particles `q`, and horizon `H`, the return gradient is estimated through the Gaussian transition log likelihood:

\[
\widehat g_b(a)=\frac1Q\sum_{q=1}^Q\sum_{t=0}^{H-1}G_{q,t}\,\nabla_{\theta_b}\log f_{\theta_b}(s^{(q)}_{t+1}\mid s^{(q)}_t,a_t).
\]

`G_(q,t)` contains discounted reward from **the current transition onward**, necessary because the HalfCheetah velocity reward depends on `s_(t+1)`. A leave-one-particle-out baseline reduces score-function estimator variance without changing its expectation under independent particle sampling. The mean over particles must be formed **before** the parameter-gradient norm.

`eq16_core.py` implements this via `torch.autograd.grad` with batched vector-Jacobian products. Sampling is detached in the likelihood-ratio computation, while `θ` is differentiated but not updated during planning. `epsilon=0` gives the unregularized PETS-style baseline.

## Mathematical scope and limits

1. The uncertainty coordinates are the *weights* of each dynamics NN with an **Euclidean parameter norm**. The original paper discusses a Wasserstein ambiguity set over transition models without an explicit concrete model-function ground metric. Thus this penalty does **not** by itself certify robustness against arbitrary transition-kernel shifts.
2. A Gaussian parameter gradient is not automatically an input score `∇_(s,a) log q_φ(s,a)`. The old branch calculates the latter and is an alternative heuristic.
3. Nonlinear-model linearization, curvature and finite-sample particle uncertainty matter; a positive nominal DR score is not a formal worst-case return guarantee.
4. The squared norm of a noisy Monte Carlo gradient is upward-biased. The number of particles, variance reduction and pathwise-gradient comparisons are important ablations.
5. Evaluate actual actuator, mass, friction shifts separately from NN-parameter perturbations: they are not proven equivalent.

## Tests and research use

Run `cd implementations/eq16 && python -m pytest -q tests`. Gaussian toy gradients have known analytical values; other tests cover null penalties, discrete action consistency, physics recalculation and resume/seed partitioning.

References:
- DR-PETS: https://arxiv.org/abs/2503.20660
- PETS: https://arxiv.org/abs/1805.12114
