## Instructions for setting up an environment.

We recommend running:

```bash
conda env create -f environment.yml
conda activate model_soups
```

However, one can also run:

```bash
# make a conda environment and install pytorch.
conda create --name model_soups python=3.6
conda activate model_soups
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0

# instructions from https://github.com/openai/CLIP for how to install. Also we will tie to a specific release.
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git@40f5484c1c74edd83cb9cf687c6ab92b28d8b656

# additional imports
pip install wget
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
pip install requests
pip install matplotlib
pip install pandas
```