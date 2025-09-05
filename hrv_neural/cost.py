import torch, torch.nn.functional as F

def calculate_cost(pred, target):
    return F.mse_loss(pred, target)

def demo_loss():
    pred = torch.tensor([1.0,2.0,3.0])
    target = torch.tensor([1.1,1.9,3.2])
    return calculate_cost(pred,target).item()
