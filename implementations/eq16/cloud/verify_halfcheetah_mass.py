"""Live MuJoCo verification after applying the HalfCheetah mass-shift fix.

Run on the EC2 instance after installing Gymnasium/MuJoCo:
    python cloud/verify_halfcheetah_mass.py
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import HalfCheetahTask

nominal = HalfCheetahTask(mass=1.0, seed=0)
shifted = HalfCheetahTask(mass=1.3, seed=0)
try:
    m0 = nominal.env.unwrapped.model
    m1 = shifted.env.unwrapped.model
    np.testing.assert_allclose(m1.body_mass, 1.3 * m0.body_mass, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(m1.body_inertia, 1.3 * m0.body_inertia, rtol=1e-6, atol=1e-7)
    # This is a compiled MuJoCo model constant (rather than an input field).
    # It should be recomputed by mj_setConst after mass perturbations.
    np.testing.assert_allclose(m1.body_subtreemass, 1.3 * m0.body_subtreemass, rtol=1e-6, atol=1e-7)
    print('PASS: 1.3x mass/inertia scaling and recomputed body_subtreemass')
    print('Nominal total mass:', float(np.sum(m0.body_mass)))
    print('Shifted total mass:', float(np.sum(m1.body_mass)))
finally:
    nominal.close()
    shifted.close()
