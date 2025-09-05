import torch, torch.nn as nn
from torchdiffeq import odeint

class GRUDynamics(nn.Module):
    def __init__(self, hrv_dim=6, latent_dim=32):
        super().__init__()
        self.gru = nn.GRU(hrv_dim, latent_dim, batch_first=True)

    def forward(self, seq, h0=None):
        out,_ = self.gru(seq,h0)
        return out[:,-1]

class ODEFunc(nn.Module):
    def __init__(self, hrv_dim=6, latent_dim=32):
        super().__init__()
        self.lin = nn.Linear(latent_dim+hrv_dim, latent_dim)
    def forward(self, t, state):
        s,h = state
        ds = torch.tanh(self.lin(torch.cat([s,h],dim=-1)))
        dh = torch.zeros_like(h)
        return ds, dh

class NeuralODEDynamics(nn.Module):
    def __init__(self, hrv_dim=6, latent_dim=32):
        super().__init__()
        self.func = ODEFunc(hrv_dim,latent_dim)
    def forward(self, seq, s0):
        s = s0
        for h in seq.unbind(dim=1):
            s,_ = odeint(self.func,(s,h),(torch.tensor(0.),torch.tensor(1.)))
            s = s[-1]
        return s

def demo():
    seq = torch.randn(4,10,6)
    model = GRUDynamics()
    print('GRU latent:', model(seq).shape)
