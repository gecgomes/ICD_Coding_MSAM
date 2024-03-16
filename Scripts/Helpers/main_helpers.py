import torch
import random
import numpy as np
from typing import Optional,Tuple
from config import *
from condicional_config import *
import pickle


def configure_seed(seed):
    """
    Set deterministic seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def compute_max_length(encodings):
    if "LE" in MODEL_TYPE:
        lengths = list(range(MIN_TEXT_LENGTH,MAX_TEXT_LENGTH+1,MIN_TEXT_LENGTH))
    elif "CE" in MODEL_TYPE:
        lengths = [MIN_TEXT_LENGTH] + list(range(1021,MAX_TEXT_LENGTH,510))
    num_tokens = len(encodings.input_ids)
    if num_tokens <= min(lengths):
        max_length = min(lengths)
    elif num_tokens > max(lengths):
        max_length = max(lengths)
    else:
        max_length = num_tokens
        for n in lengths:
            if max_length <= n:
                max_length = n
                break        
    return max_length 

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def free_tensor(tensor):
    tensor.detach()
    tensor.grad = None
    tensor.storage().resize_(0)
    del tensor