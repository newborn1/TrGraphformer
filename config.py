import os
import torch


class Config:
    retrain = True
    max_seqlen = 8
    n_layer = 5
    embedding_size = [32]
    output_size = 2
    dropout_prob = 0.5
    src_mask = None