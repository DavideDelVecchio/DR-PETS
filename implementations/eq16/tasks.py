"""Control task adapters. HalfCheetah requires `pip install 'gymnasium[mujoco]'`."""
from __future__ import annotations
import numpy as np
import torch
from drpets_benchmark import make_env as make_small_env

class SmallTask:
    def __init__(self,name,shift=1.0,seed=0):
        self.env=make_small_env(name,shift,seed)
        self.name=name
        self.obs_dim=self.env.dim; self.act_dim=self.env.act_dim
        self.discrete=self.env.discrete
        self.action_limit=1.0 if self.discrete else 2.0
    def reset(self,seed=None):
        if seed is not None:self.env.rng=np.random.default_rng(int(seed))
        return self.env.reset()
    def step(self,action):
        ns,r,done=self.env.step(action)
        return ns,float(r),bool(done)
    def reward_model(self,s,a,snext):
        # Original small control tasks have pre-transition rewards.
        return self.env.reward(s,a)
    def from_planner_action(self,a):return self.env.action(float(a[0]))
    def random_action(self,rng):
        return np.array([float(rng.integers(0,2)) if self.discrete else rng.uniform(-2,2)],dtype=np.float32)
    def close(self):pass

class HalfCheetahTask:
    """Official Gymnasium HalfCheetah-v5 with root x included in observation.

    This is critical: Gymnasium's default 17D observation hides root x, although
    its reward depends on (x_after-x_before)/dt. We use 18D observations.
    """
    def __init__(self,mass=1.,friction=1.,actuation=1.,seed=0):
        try:import gymnasium as gym
        except ImportError as e:raise ImportError("Install `pip install 'gymnasium[mujoco]'` for HalfCheetah-v5") from e
        self.env=gym.make('HalfCheetah-v5',exclude_current_positions_from_observation=False)
        self.name='halfcheetah';self.discrete=False
        self.obs_dim=int(self.env.observation_space.shape[0]);self.act_dim=int(self.env.action_space.shape[0])
        self.action_limit=1.
        self.dt=float(self.env.unwrapped.dt)
        assert self.obs_dim==18, f'Expected 18D including root position, got {self.obs_dim}'
        # The environment is freshly created for this physical-parameter setting,
        # so shifts always apply to the official nominal model (never cumulatively).
        # Changing compiled MuJoCo body mass/inertia requires rebuilding constants.
        # Keep the official XML unchanged: `settotalmass` is deprecated but still
        # defines the correct nominal baseline for Gymnasium HalfCheetah-v5.
        import mujoco
        sim=self.env.unwrapped.model
        data=self.env.unwrapped.data
        sim.body_mass[:]*=float(mass)
        sim.body_inertia[:]*=float(mass)
        sim.geom_friction[:,0]*=float(friction)
        sim.actuator_gear[:,0]*=float(actuation)
        mujoco.mj_setConst(sim, data)
        # mj_setConst uses mjData as scratch space; reset before simulation.
        self.env.reset(seed=seed)
    def reset(self,seed=None):
        state,_=self.env.reset(seed=seed)
        return np.asarray(state,dtype=np.float32)
    def step(self,action):
        state,reward,terminated,truncated,_=self.env.step(np.asarray(action,dtype=np.float32))
        return np.asarray(state,dtype=np.float32),float(reward),bool(terminated or truncated)
    def reward_model(self,s,a,snext):
        return (snext[...,0]-s[...,0])/self.dt - 0.1*a.square().sum(-1)
    def from_planner_action(self,a):return np.clip(a,-1.,1.).astype(np.float32)
    def random_action(self,rng):return rng.uniform(-1,1,size=self.act_dim).astype(np.float32)
    def close(self):self.env.close()


def make_task(name,shift,seed):
    if name=='halfcheetah':
        return HalfCheetahTask(mass=shift.get('mass',1.),friction=shift.get('friction',1.),
                              actuation=shift.get('actuation',1.),seed=seed)
    return SmallTask(name,float(shift),seed)
