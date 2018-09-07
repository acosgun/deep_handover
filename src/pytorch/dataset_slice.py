#!/usr/bin/env python
from torch.utils.data import Dataset

class DataSlice(Dataset):
    def __init__(self,dataset,start,end):
        self.dataset = dataset
        self.start = start
        self.end = end


    def __len__(self):
        return self.end-self.start


    def __getitem__(self,index):
        index -= self.start
        return self.dataset[index]
