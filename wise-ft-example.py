import argparse
import os
import wget
import torch
import clip
import os

from datasets import ImageNet, ImageNetR
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
        "--alpha",
        type=float,
        default=0.75,
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

    if args.download_models:

        # download the ViT-L/14 0-shot model
        wget.download(
            f'https://github.com/mlfoundations/model-soups/releases/download/v0.0.3/zero-shot-vit-l14.pt',
            out=args.model_location
        )

        # download the ViT-L/14 fine-tuned model model
        wget.download(
            f'https://github.com/mlfoundations/model-soups/releases/download/v0.0.3/checkpoint-4-vit-l14.pt',
            out=args.model_location
        )

    base_model, preprocess = clip.load('ViT-L/14', 'cpu', jit=False)

    # load the state dictionary for the zero-shot and fine-tuned model state dicts (sds)
    zero_shot_sd = torch.load(os.path.join(args.model_location, 'zero-shot-vit-l14.pt'), map_location=torch.device('cpu'))
    ft_sd = torch.load(os.path.join(args.model_location, 'checkpoint-4-vit-l14.pt'), map_location=torch.device('cpu'))

    # interpolate with coefficient alpha
    wise_ft_sd = {k : zero_shot_sd[k] * (1 - args.alpha) + ft_sd[k] * args.alpha for k in zero_shot_sd}


    for sd in [zero_shot_sd, ft_sd, wise_ft_sd]:
        model = get_model_from_sd(sd, base_model)

        for dataset_cls in [ImageNet, ImageNetR]:
            dataset = dataset_cls(preprocess, args.data_location, args.batch_size, args.workers)
            print(f'Evaluation on {dataset_cls.__name__}')
            accuracy = test_model_on_dataset(model, dataset)
            print(f"Accuracy on {dataset_cls.__name__}: {accuracy * 100}")

    """
    ImageNet:
    CLIP ViT-L zero-shot: 75.5
    CLIP ViT-L fine-tuned: 85.5
    WiSE-FT (0.75): 86.0

    ImageNet-R:
    CLIP ViT-L zero-shot: 87.8
    CLIP ViT-L fine-tuned: 84.3
    WiSE-FT (0.75): 87.4
    """