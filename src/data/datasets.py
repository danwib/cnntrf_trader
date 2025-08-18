import numpy as np, torch
from torch.utils.data import Dataset
class WindowedDataset(Dataset):
    def __init__(self,X,y,seq_len): self.X=X; self.y=y; self.seq_len=seq_len
    def __len__(self): return len(self.X)-self.seq_len
    def __getitem__(self,i): import torch; w=self.X[i:i+self.seq_len].T; t=self.y[i+self.seq_len-1]; return torch.tensor(w).float(), torch.tensor([t]).float()
