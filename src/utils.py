
import torch

def collate_fn(batch):
    return tuple(zip(*batch))

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
