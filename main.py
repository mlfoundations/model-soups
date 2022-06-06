import argparse
import os
import wget
import torch
import clip
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA
from utils import get_model_from_sd, test_model_on_dataset

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
        "--eval-individual-models", action="store_true", default=False,
    )
    parser.add_argument(
        "--uniform-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--greedy-soup", action="store_true", default=False,
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
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



if __name__ == '__main__':
    args = parse_arguments()
    NUM_MODELS = 72
    INDIVIDUAL_MODEL_RESULTS_FILE = 'individual_model_results.jsonl'
    UNIFORM_SOUP_RESULTS_FILE = 'uniform_soup_results.jsonl'

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


    # Step 2: Evaluate individual models.
    if args.eval_individual_models or args.uniform_soup:
        base_model, preprocess = clip.load('ViT-B/32', 'cpu', jit=False)

    if args.eval_individual_models:
        if os.path.exists(INDIVIDUAL_MODEL_RESULTS_FILE):
            os.remove(INDIVIDUAL_MODEL_RESULTS_FILE)
        for j, model_path in enumerate(model_paths):
            assert os.path.exists(model_path)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model = get_model_from_sd(state_dict, base_model)

            results = {'model_name' : f'model_{j}'}
            # Note: ImageNet2p is the held-out minival set from ImageNet train that we use.
            # It is called 2p for 2 percent of ImageNet, or 26k images.
            # See utils on how this dataset is handled slightly differently.
            for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:

                print(f'Evaluating model {j} of {NUM_MODELS - 1} on {dataset_cls.__name__}.')

                dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
                accuracy = test_model_on_dataset(model, dataset)
                results[dataset_cls.__name__] = accuracy
                print(accuracy)

            with open(INDIVIDUAL_MODEL_RESULTS_FILE, 'a+') as f:
                f.write(json.dumps(results) + '\n')

    # Step 3: Uniform Soup.
    if args.uniform_soup:
        if os.path.exists(UNIFORM_SOUP_RESULTS_FILE):
            os.remove(UNIFORM_SOUP_RESULTS_FILE)
        uniform_soup_location = os.path.join(args.model_location, 'uniform_soup.pt')
        if os.path.exists(uniform_soup_location):
            os.remove(uniform_soup_location)

        # create the uniform soup sequentially to not overload memory
        for j, model_path in enumerate(model_paths):

            print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

            assert os.path.exists(model_path)
            state_dict = torch.load(model_path)
            if j == 0:
                uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
            else:
                uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

        model = get_model_from_sd(uniform_soup, base_model)

        results = {'model_name' : f'uniform_soup'}
        for dataset_cls in [ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA]:

            print(f'Evaluating on {dataset_cls.__name__}.')

            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            accuracy = test_model_on_dataset(model, dataset)
            results[dataset_cls.__name__] = accuracy
            print(accuracy)
       
        with open(UNIFORM_SOUP_RESULTS_FILE, 'a+') as f:
            f.write(json.dumps(results) + '\n')


    # Step 4: Greedy Soup.


    # Step 5: Plot.
    if args.plot:
        individual_model_db = pd.read_json(INDIVIDUAL_MODEL_RESULTS_FILE, lines=True)
        individual_model_db['OOD'] = 1./5 * (individual_model_db['ImageNetV2'] + 
            individual_model_db['ImageNetR'] + individual_model_db['ImageNetSketch'] + 
            individual_model_db['ObjectNet'] + individual_model_db['ImageNetA'])
        uniform_soup_db = pd.read_json(UNIFORM_SOUP_RESULTS_FILE, lines=True)
        uniform_soup_db['OOD'] = 1./5 * (uniform_soup_db['ImageNetV2'] + 
            uniform_soup_db['ImageNetR'] + uniform_soup_db['ImageNetSketch'] + 
            uniform_soup_db['ObjectNet'] + uniform_soup_db['ImageNetA'])

        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        ax = fig.subplots()

        ax.scatter(
            uniform_soup_db['ImageNet'], 
            uniform_soup_db['OOD'], 
            marker='o', 
            color='C0',
            s=200,
            label='Uniform Soup',
            zorder=1
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[0], 
            individual_model_db['OOD'].values[0], 
            marker='h', 
            color='slategray',
            s=150,
            label='Initialization (LP)',
            zorder=1
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[1:], 
            individual_model_db['OOD'].values[1:], 
            marker='d', 
            color='C2',
            s=130,
            label='Various hyperparameters',
            zorder=1
        )

        ax.set_ylabel('Avg. accuracy on 5 distribution shifts', fontsize=16)
        ax.set_xlabel('ImageNet Accuracy (top-1%)', fontsize=16)
        ax.grid()
        ax.legend(fontsize=13)
        plt.savefig('figure.png', bbox_inches='tight')