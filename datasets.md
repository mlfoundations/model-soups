## Instructions for downloading datasets (modified from https://github.com/mlfoundations/wise-ft)

### Step 1: Download

```bash
export DATA_LOCATION=~/data # feel free to change.
cd $DATA_LOCATION
```

#### [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)

```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar xvf imagenet-a.tar
rm imagenet-a.tar
```

#### [ImageNet-R](https://github.com/hendrycks/imagenet-r)

```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar xvf imagenet-r.tar
rm imagenet-r.tar
```

#### [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

Download links:
- from [Google Drive](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA)
- from [Kaggle](https://www.kaggle.com/wanghaohan/imagenetsketch)

#### [ImageNet V2](https://github.com/modestyachts/ImageNetV2)

```bash
wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz
rm imagenetv2-matched-frequency.tar.gz
```

#### [ObjectNet](https://objectnet.dev/)

```bash
wget https://objectnet.dev/downloads/objectnet-1.0.zip
unzip objectnet-1.0.zip
rm objectnet-1.0.zip
```

#### ImageNet

Can be downloaded via https://www.image-net.org/download.php.
Please format for PyTorch, e.g., via https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh.

### Step 2: Check that datasts are downloaded

When running:
```bash
cd $DATA_LOCATION
ls
```
you should see (at least):
```bash
imagenet # containing train and val subfolders
imagenetv2-matched-frequency-format-val
imagenet-r
imagenet-a
sketch # imagenet-sketch
objectnet-1.0
```
