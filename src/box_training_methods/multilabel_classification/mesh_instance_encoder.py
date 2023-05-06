from typing import *

import torch
import wandb
from torch import Tensor, LongTensor
from torch.nn import Module, Parameter
from torch.nn import functional as F
from wandb_utils.loggers import WandBLogger

from transformers import BioGptModel

from box_training_methods.models.temps import convert_float_to_const_temp
from box_training_methods.utils import tiny_value_of_dtype
from box_training_methods import metric_logger

__all__ = [
    "MeshInstanceEncoder",
]


class MeshInstanceEncoder(Module):
    
    def __init__(self, output_dim):
        super().__init__()

        self.model = BioGptModel.from_pretrained("microsoft/biogpt")
        self.output_dim = output_dim

        self.proj = torch.nn.Linear(1024, self.output_dim)  # TODO model-dependent dim
        self.delta = torch.tensor(1.0e-5, requires_grad=False)

    def forward(self, inputs):
        inputs = {
            'input_ids': inputs['input_ids'].to(self.model.device), 
            'attention_mask': inputs['attention_mask'].to(self.model.device)
        }
        outputs = self.model(**inputs)  # (batch_size, max_seq_len, dim)
        # TODO access CLS token specifically (c.f. CollateMeshFn)
        x = outputs.last_hidden_state[:,-1,:]  # (batch_size, dim)
        instance_min = self.proj(x)  # (batch_size, output_dim)
        instance_max = instance_min + self.delta
        instance_box = torch.stack([instance_min, -instance_max], dim=-2)
        return instance_box
