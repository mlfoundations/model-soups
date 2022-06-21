import argparse
import os
import torch
import clip
import os
from tqdm import tqdm

import datasets
from utils import ModelWrapper, test_model_on_dataset
from openai_imagenet_template import openai_imagenet_template

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
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--custom-template", action="store_true", default=False,
    )
    parser.add_argument(
        "--dataset",  default="ImageNet", 
        help=f"Must be one of {','.join(['ImageNet', 'ImageNetV2', 'ImageNetR', 'ObjectNet', 'ImageNetA'])}"

    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--model",
        default='ViT-B/32',
        help='Model to use -- you can try another like ViT-L/14'
    )
    return parser.parse_args()

def zeroshot_classifier(model, classnames, templates, device):
    print('Building zero-shot classifier.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return 100*zeroshot_weights.t()


if __name__ == '__main__':
    args = parse_arguments()
    DEVICE = 'cuda'
    assert args.dataset in ['ImageNet', 'ImageNetV2', 'ImageNetR', 'ObjectNet', 'ImageNetA']

    if args.custom_template:
        template = [lambda x : f"a photo of a {x}."]
    else:
        template = openai_imagenet_template

    base_model, preprocess = clip.load(args.model, 'cuda', jit=False)
    dset = getattr(datasets, args.dataset)(preprocess, location=args.data_location, batch_size=args.batch_size, num_workers=args.workers)
    clf = zeroshot_classifier(base_model, dset.classnames, template, DEVICE)
    NUM_CLASSES = len(dset.classnames)
    feature_dim = base_model.visual.output_dim

    model = ModelWrapper(base_model, feature_dim, NUM_CLASSES, normalize=True, initial_weights=clf)
    for p in model.parameters():
        p.data = p.data.float()

    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model,  device_ids=devices)

    accuracy = test_model_on_dataset(model, dset)

    print(f'Accuracy is {round(100 * accuracy, 2)}.')
