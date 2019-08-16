from collections import OrderedDict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

class BaseOperation(object):
    def __init__(self, model):
        super(BaseOperation, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = [] 

    def gen_onehot_encoding(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.model.zero_grad()
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        one_hot = self.gen_onehot_encoding(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()
 
