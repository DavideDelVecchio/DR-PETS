import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from torch import nn
from eq16_core import eq16_scores, RolloutConfig, GaussianDynamics

class ToyGaussian(nn.Module):
    # x_next=x+theta+sigma*noise, reward=x_next. Analytic gradient for H
    # transitions: H(H+1)/2 (the reward at t is x_{t+1}).
    def __init__(self,mu=0.2,sigma=1.0):
        super().__init__();self.mu=nn.Parameter(torch.tensor(mu));self.sigma=sigma
    def sample_and_score(self,x,a,generator):
        noise=torch.randn(x.shape,device=x.device,generator=generator)
        nx=(x.detach()+self.mu+self.sigma*noise).detach()
        p=torch.distributions.Normal(self.mu,self.sigma)
        lp=p.log_prob((nx-x.detach()).detach()).sum(-1)
        return nx,lp

def reward(s,a,nxt):return nxt[...,0]

@pytest.mark.parametrize('horizon,truth',[(1,1.0),(2,3.0),(3,6.0)])
def test_score_gradient_matches_analytic(horizon,truth):
    model=ToyGaussian();initial=torch.zeros(1);acts=torch.zeros(1,horizon,1)
    conf=RolloutConfig(horizon=horizon,particles=15000,gamma=1.,chunk=1)
    val,nom,pen=eq16_scores([model],initial,acts,reward,conf,epsilon=.2,seed=42)
    assert abs(pen.item()-truth)<.18, (horizon,pen.item(),truth)
    assert torch.allclose(val,nom-.2*pen,atol=1e-5)
    assert abs(model.mu.item()-.2)<1e-7


def test_constant_reward_zero_score():
    model=ToyGaussian();initial=torch.zeros(1);acts=torch.zeros(1,3,1)
    constant=lambda x,a,nx: torch.ones_like(nx[...,0])
    val,nom,pen=eq16_scores([model],initial,acts,constant,
                             RolloutConfig(horizon=3,particles=2000,chunk=1),epsilon=.3)
    assert pen.item()<1e-5


def test_epsilon_zero_equals_nominal_and_does_not_change_weights():
    model=ToyGaussian();initial=torch.zeros(1);acts=torch.zeros(4,2,1)
    val,nom,pen=eq16_scores([model],initial,acts,reward,
                             RolloutConfig(horizon=2,particles=100,chunk=2),epsilon=0)
    assert torch.allclose(val,nom)
    assert torch.all(pen==0)
    assert abs(model.mu.item()-.2)<1e-7


def test_gaussian_dynamics_has_learned_mean_variance():
    m=GaussianDynamics(3,2,width=8,depth=2)
    x=torch.zeros(4,3);a=torch.zeros(4,2)
    mu,logstd=m(x,a)
    assert mu.shape==(4,3) and logstd.shape==(4,3)
    assert torch.isfinite(logstd).all()
