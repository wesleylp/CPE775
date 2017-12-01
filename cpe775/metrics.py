import torch

def rmse(y_pred, y_true):
    return torch.sqrt(torch.mean(y_pred-y_true)**2)
