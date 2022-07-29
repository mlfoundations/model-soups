###
# Disclaimer: This is not our central method, we recommend the greedy soup which is found in main.py.
# This method is described in appendix I and, compared to main.py, this code is much less tested.
# For instance, we don't know how stable the results are under optimization noise. However, we expect
# this method to outperform greedy soup. Still, we recommend using greedy soup and not this. 
# As mentioned in the paper, this code is computationally expernsive as it requires loading models in memory.
# We run this on a node with 490GB RAM and use 1 GPU with 40GB of memory.
# It also looks like PyTorch released a very helpful utility which we recommend if re-implementing:
# https://pytorch.org/docs/stable/generated/torch.nn.utils.stateless.functional_call.html?utm_source=twitter&utm_medium=organic_social&utm_campaign=docs&utm_content=functional-api-for-modules
# When running with lr = 0.05 and epochs = 5 we get 81.38%.
###

import argparse
import os
import wget
import torch
import clip
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.utils.data import SubsetRandomSampler
from torchvision import models
from torch.autograd.functional import vhp, jvp, jacobian
from torchvision import datasets

from datasets.imagenet import ImageNet2pShuffled, ImageNet

from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/ssd/checkpoints/soups'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--download-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    return parser.parse_args()

# Utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])
def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class AlphaWrapper(torch.nn.Module):
    def __init__(self, paramslist, model, names):
        super(AlphaWrapper, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        ralpha = torch.ones(len(paramslist[0]), len(paramslist))
        ralpha = torch.nn.functional.softmax(ralpha, dim=1)
        self.alpha_raw = torch.nn.Parameter(ralpha)
        self.beta = torch.nn.Parameter(torch.tensor(1.))

    def alpha(self):
        return torch.nn.functional.softmax(self.alpha_raw, dim=1)

    def forward(self, inp):
        alph = self.alpha()
        params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        out = self.model(inp)
        return self.beta * out

def get_imagenet_acc(test_dset):
    with torch.no_grad():
        correct = 0.
        n = 0
        end = time.time()
        for i, batch in enumerate(test_dset.test_loader):
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()
            data_time = time.time() - end
            end = time.time()
            logits = alpha_model(inputs)
            loss = criterion(logits, labels)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            y = labels
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            batch_time = time.time() - end
            percent_complete = 100.0 * i / len(test_dset.test_loader)
            if ( i % 10 ) == 0:
                print(
                    f"Train Epoch: {0} [{percent_complete:.0f}% {i}/{len(test_dset.test_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
            end = time.time()
        acc = correct / float(n)
        print('Top-1', acc)
    return acc


if __name__ == '__main__':
    args = parse_arguments()
    NUM_MODELS = 72

    # Step 1: Download models.
    if args.download_models:
        if not os.path.exists(args.model_location):
            os.mkdir(args.model_location)
        for i in range(NUM_MODELS):
            print(f'\nDownloading model {i} of {NUM_MODELS - 1}')
            wget.download(
                f'https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{i}.pt',
                out=args.model_location
                )

    model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_MODELS)]
    base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda')
    train_dset = ImageNet2pShuffled(preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    test_dset = ImageNet(preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)

    sds = [torch.load(cp, map_location='cpu') for cp in model_paths]
    feature_dim = sds[0]['classification_head.weight'].shape[1]
    num_classes = sds[0]['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    model = model.to(device)

    _, names = make_functional(model)
    first = False

    paramslist = [tuple(v.detach().requires_grad_().cpu() for _, v in sd.items()) for i, sd in enumerate(sds)]
    torch.cuda.empty_cache()
    alpha_model = AlphaWrapper(paramslist, model, names)


    print(alpha_model.alpha())
    print(len(list(alpha_model.parameters())))

    lr = 0.05
    epochs = 5

    optimizer = torch.optim.AdamW(alpha_model.parameters(), lr=lr, weight_decay=0.)
    num_batches = len(train_dset.train_loader)

    for epoch in range(epochs):
        end = time.time()
        for i, batch in enumerate(train_dset.train_loader):
            step = i + epoch * num_batches
            batch = maybe_dictionarize_batch(batch)
            inputs, labels = batch['images'].cuda(), batch['labels'].cuda()

            data_time = time.time() - end
            end = time.time()

            optimizer.zero_grad()

            out = alpha_model(inputs)

            loss = criterion(out, labels)
            
            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            percent_complete = 100.0 * i / len(train_dset.train_loader)
            if ( i % 10 ) == 0:
                # print(alpha_model.beta)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_dset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
                # print(alpha_model.alpha())
            end = time.time()

    acc = get_imagenet_acc(test_dset)
    print('Accuracy is', 100 * acc)

    # torch.save(
    #     {'alpha' : alpha_model.alpha(), 'beta' : alpha_model.beta}, 
    #     f'alphas_{lr}_{epochs}.pt'
    # )
