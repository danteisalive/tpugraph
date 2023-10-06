import os
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import  DataLoader

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.pytorch_utils import apply_chunking_to_forward
from transformers.activations import ACT2FN
import pytorch_lightning as pl
import torchmetrics as tm
# import bitsandbytes as bnb