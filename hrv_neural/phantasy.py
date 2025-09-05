import torch, torch.nn as nn

class MultimodalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(3,64),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,2)
        )
    def forward(self,x):
        return self.net(x)

def demo():
    gen=MultimodalGenerator()
    sample=torch.randn(5,3)
    print('Generator output (sound/light):',gen(sample))
