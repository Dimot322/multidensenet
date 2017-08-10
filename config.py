import os
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models

from bce import binary_cross_entropy_with_logits
from densenet import DenseNetBase, DenseNet121, DenseNet100
from multilabel_dataset import MultiLabelDataset

def get_loss_function(multilabel):
    if multilabel:
        return binary_cross_entropy_with_logits
    return F.cross_entropy

def get_dataset(name, partition, transform):
    image_dir = os.path.join('input', '%s-%s' % (name, partition))
    if name == 'imagenet':
        return datasets.ImageFolder(image_dir, transform)
    elif name == 'food-collage':
        csv_path = os.path.join('input', '%s-%s.csv' % (name, partition))
        return MultiLabelDataset(csv_path, image_dir, transform) 

def get_network(name, num_classes, pretrained):
    if name == 'densenet-base':
        assert not pretrained
        return DenseNetBase(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=num_classes)
    elif name == 'densenet-121':
        return DenseNet121(num_classes, pretrained) 
    elif name == 'densenet-100':
        assert not pretrained
        return DenseNet100(num_classes)
