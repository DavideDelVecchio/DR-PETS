"""Verify that MuJoCo model constants are rebuilt after physics perturbations.

Uses a minimal fake Gymnasium/MuJoCo interface so the regression test also runs
on CPU nodes without the optional MuJoCo dependencies installed.
"""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import HalfCheetahTask


@pytest.mark.parametrize(
    "mass,friction,actuation",
    [(1.0, 1.0, 1.0), (0.7, 1.5, 0.8), (1.3, 0.5, 1.1)],
)
def test_model_constants_recomputed_before_reset(monkeypatch, mass, friction, actuation):
    calls = []
    model = SimpleNamespace(
        body_mass=np.array([0.0, 2.0, 3.0]),
        body_inertia=np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        geom_friction=np.array([[0.4, 0.1, 0.1], [0.7, 0.2, 0.2]]),
        actuator_gear=np.array([[50.0, 0.0], [70.0, 0.0]]),
    )
    originals = {k: getattr(model, k).copy() for k in vars(model)}
    data = object()

    class Env:
        unwrapped = None
        observation_space = SimpleNamespace(shape=(18,))
        action_space = SimpleNamespace(shape=(6,))
        dt = 0.05

        def __init__(self):
            self.unwrapped = self
            self.model = model
            self.data = data

        def reset(self, *, seed=None):
            calls.append(('reset', seed))
            return np.zeros(18, dtype=np.float32), {}

        def close(self):
            calls.append(('close',))

    fake_env = Env()
    gym = ModuleType('gymnasium')
    gym.make = lambda name, **kwargs: (
        fake_env if name == 'HalfCheetah-v5' and
        kwargs.get('exclude_current_positions_from_observation') is False
        else (_ for _ in ()).throw(AssertionError('unexpected environment'))
    )
    mujoco = ModuleType('mujoco')

    def fake_set_const(given_model, given_data):
        calls.append(('setConst',))
        assert given_model is model
        assert given_data is data
        # Assert the new model values are present at the moment of recomputation.
        np.testing.assert_allclose(model.body_mass, originals['body_mass'] * mass)
        np.testing.assert_allclose(model.body_inertia, originals['body_inertia'] * mass)
        np.testing.assert_allclose(model.geom_friction[:, 0], originals['geom_friction'][:, 0] * friction)
        np.testing.assert_allclose(model.actuator_gear[:, 0], originals['actuator_gear'][:, 0] * actuation)

    mujoco.mj_setConst = fake_set_const
    monkeypatch.setitem(sys.modules, 'gymnasium', gym)
    monkeypatch.setitem(sys.modules, 'mujoco', mujoco)

    task = HalfCheetahTask(mass=mass, friction=friction, actuation=actuation, seed=123)
    try:
        assert calls == [('setConst',), ('reset', 123)]
        assert task.obs_dim == 18 and task.act_dim == 6
    finally:
        task.close()
