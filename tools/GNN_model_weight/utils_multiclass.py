import os
from typing import Union
import math

import yaml
import uproot
import awkward as ak
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import IterableDataset, get_worker_info
from scipy.stats import entropy, gaussian_kde

from ..GNN_model_weight.models import mdn_loss, mdn_loss_new

class GraphIterableDataset(IterableDataset):
    def __init__(self, data_dir, limit_files=None):
        super().__init__()
        self.data_dir = data_dir
        self.files = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
        )
        if limit_files is not None:
            self.files = self.files[:limit_files]

        self.num_files = len(self.files)
        self.num_graphs = 0  # Will be set when iterating
        self.class_counts = None  # Will be set when iterating
        self.class_weights = None  # Will be set when iterating

    def __len__(self):
        if self.num_graphs != 0:
            return self.num_graphs
        else:
            self.class_counts = {} # Initialize the dictionary
            self.class_weights = {}
            for file in self.files:
                dataset = torch.load(file, weights_only=False)
                self.num_graphs += len(dataset)
                for graph in dataset:
                    label = int(graph.y)
                    self.class_counts[label] = self.class_counts.get(label, 0) + 1
                    self.class_weights[label] = self.class_weights.get(label, 0.0) + float(graph.weight)
                del dataset
        return self.num_graphs # Return number of files
    
    def get_class_counts(self):
        if self.class_counts is None:
            self.__len__()  # This will populate class_counts
        return self.class_counts
    
    def get_class_weights(self):
        if self.class_weights is None:
            self.__len__()  # This will populate class_weights
        return self.class_weights

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading, return full iterator
            files = self.files
        else:
            # Split workload between workers
            per_worker = int(math.ceil(len(self.files) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.files))
            files = self.files[start:end]

        for file in files:
            # print(f"Worker loading file: {file}")
            dataset = torch.load(file, weights_only=False)  # list of Data objects
            for graph in dataset:
                yield graph

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def train_multi(loader, model, device, optimizer, epoch):
    # print("dataset files:", len(dataset.files))
    model.train()
    loss_all = 0
    batch_counter = 0

    for data in loader:  # now directly over the iterable dataset
        batch_counter += 1
        # print("batch_counter:", batch_counter, end="\r")

        # print(f'batch {batch_counter} length:', len(data))

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)  # shape [batch_size, 4]
        # Make sure labels are tensor and integer type
        targets = torch.as_tensor(data.y, dtype=torch.long, device=device) - 1  

        # Make sure weights are tensor float type
        sample_weights = torch.as_tensor(data.weight, dtype=torch.float, device=device)

        loss_per_sample = F.nll_loss(output, targets, reduction='none') # softmax_loss = log_softmax + nll_loss

        loss_scalar = (loss_per_sample * sample_weights).mean()
        loss_scalar.backward() # Calculate gradients
        optimizer.step() # Update weights

        loss_all += data.num_graphs * loss_scalar.item()

    torch.cuda.empty_cache()
    return loss_all / len(loader.dataset)


@torch.no_grad()
def get_accuracy_multi(loader, model, device):
    #remember to change this when evaluating combined model
    model.eval()
    correct = 0
    for data in loader:
        cl_data = data.to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        output = model(cl_data)
        pred = F.Softmax(output).max(dim=1)
        correct += pred.eq(new_y[0,:]).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def test_multi(loader, model, device):
    model.eval()
    #print("init my_test()")
    #time.sleep(600)
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        #print("batch_counter: ",batch_counter, end="\r")
        data = data.to(device)
        output = model(data)  # shape [batch_size, 4]
        # Make sure labels are tensor and integer type
        # print('data.y:', data.y)
        targets = torch.as_tensor(data.y, dtype=torch.long, device=device) - 1  

        # Make sure weights are tensor float type
        sample_weights = torch.as_tensor(data.weight, dtype=torch.float, device=device)
        loss_per_sample = F.nll_loss(output, targets, reduction='none')
        loss_scalar = (loss_per_sample * sample_weights).mean()
        loss_all += data.num_graphs * loss_scalar.item()

    del data
    data = []
    torch.cuda.empty_cache()
    return loss_all/len(loader.dataset)

@torch.no_grad()
def get_scores_multi(loader, model, device, class_n = 4):
    model.eval()
    total_output = np.zeros((1, class_n)) # Initialize the 2d array
    # total_output = np.array([[1]])
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        # print ("Processing batch", batch_counter, "of",len(loader))
        data = data.to(device)
        pred = model(data)
        total_output = np.append(total_output, pred.cpu().detach().numpy(), axis=0)

    return total_output[1:]