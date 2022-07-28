import os
import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np

from .common import ImageFolderWithPaths, SubsetSampler
from .imagenet_classnames import get_classnames

class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai',
                 distributed=False):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.distributed = distributed

        self.populate_train()
        self.populate_test()
    
    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess,
            )
        sampler = self.get_train_sampler()
        self.sampler = sampler
        kwargs = {'shuffle' : True} if sampler is None else {}
        # print('kwargs is', kwargs)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_train_sampler(self):
        return torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.distributed else None


    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return 'imagenet'

class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass

def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)

class ImageNetSubsample(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)

class ImageNetSubsampleValClasses(ImageNet):
    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass
    
    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])
        
        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [self.class_sublist.index(int(label)) for label in labels]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


class ImageNet98p(ImageNet):

    def get_train_sampler(self):
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        #if os.path.exists(idx_file):
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)
        # else:
        #     idxs = np.zeros(len(self.train_dataset.targets))
        #     target_array = np.array(self.train_dataset.targets)
        #     for c in range(1000):
        #         m = target_array == c
        #         n = len(idxs[m])
        #         arr = np.zeros(n)
        #         arr[:26] = 1
        #         np.random.shuffle(arr)
        #         idxs[m] = arr
        #     with open(idx_file, 'wb') as f:
        #         np.save(f, idxs)

        idxs = (1 - idxs).astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])

        return sampler


class ImageNet2p(ImageNet):

    def get_train_sampler(self):
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)

        idxs = idxs.astype('int')
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler

class ImageNet2pShuffled(ImageNet):

    def get_train_sampler(self):
        print('shuffling val set.')
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
        return sampler
