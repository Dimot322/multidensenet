#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from openimages import OpenImagesDataset

import os
import sys
import math

import shutil

import setproctitle

import densenet
import make_graph

IMG_PATH = os.path.join('input', 'collage-23')
TEST_IMG_PATH = os.path.join('input', 'test-collage-23', 'images')
TRAIN_DATA = os.path.join('input', 'collage-23.csv')
VAL_DATA = os.path.join('input', 'val.csv')
TEST_DATA = os.path.join('input', 'test-collage-23', 'test.csv')
thresh = 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=50)
    parser.add_argument('--sEpoch', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/densenet.base.collage-23'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # if os.path.exists(args.save):
    #     shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = valTransform = transforms.Compose([
        transforms.Scale(64),
        transforms.RandomCrop(56, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.Scale(64),
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        normTransform
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        OpenImagesDataset(TRAIN_DATA, IMG_PATH, trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)
    valLoader = DataLoader(
        OpenImagesDataset(VAL_DATA, IMG_PATH, valTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        OpenImagesDataset(TEST_DATA, TEST_IMG_PATH, testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)

    net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=23)
    # net = torch.load(os.path.join(args.save, '13.pth'))
    
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'a')
    valF = open(os.path.join(args.save, 'val.csv'), 'a')
    testF = open(os.path.join(args.save, 'test.csv'), 'a')

    for epoch in range(args.sEpoch, args.nEpochs + args.sEpoch):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        # val(args, epoch, net, valLoader, optimizer, valF)
        test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, '%d.pth' % epoch))
        # os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    valF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.binary_cross_entropy(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.gt(thresh)
        # t_pred = pred.eq(1)
        # f_pred = pred.eq(0)
        # print(t_pred)
        # t = target.data.byte().eq(1)
        # print(t)
        # f = target.data.byte().eq(0)
        tp = (pred + target.data.byte()).eq(2).sum()
        fp = (pred - target.data.byte()).eq(1).sum()
        fn = (pred - target.data.byte()).eq(-1).sum()
        tn = (pred + target.data.byte()).eq(0).sum()
        # correct = pred.eq(target.data.byte()).cpu().sum()
        # acc = 100. * correct / (pred.size()[0] * pred.size()[1])
        acc = (tp + tn) / (tp + tn + fp + fn)
        try:
            prec = tp / (tp + fp)
        except ZeroDivisionError:
            prec = 0.0
        try:
            rec = tp / (tp + fn)
        except ZeroDivisionError:
            rec = 0.0
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f}\tPrec: {:.4f}\tRec: {:.4f}\tTP: {}\tFP: {}\tFN: {}\tTN: {}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], acc, prec, rec, tp, fp, fn, tn))

        trainF.write('{},{},{},{},{}\n'.format(partialEpoch, loss.data[0], acc, prec, rec))
        trainF.flush()

def val(args, epoch, net, valLoader, optimizer, valF):
    net.eval()
    best_threshold = max_rec_fpr = 0.0
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        val_loss = acc = prec = rec = fpr = 0
        for data, target in valLoader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = net(data)
            val_loss += F.binary_cross_entropy(output, target).data[0]
            pred = output.data.gt(threshold)
            tp = (pred + target.data.byte()).eq(2).sum()
            fp = (pred - target.data.byte()).eq(1).sum()
            fn = (pred - target.data.byte()).eq(-1).sum()
            tn = (pred + target.data.byte()).eq(0).sum()
            acc += (tp + tn) / (tp + tn + fp + fn)
            try:
                prec += tp / (tp + fp)
                rec += tp / (tp + fn)
                fpr += fp / (tn + fp)
            except ZeroDivisionError:
                pass
        acc /= len(valLoader)
        prec /= len(valLoader)
        rec /= len(valLoader)
        fpr /= len(valLoader)
        if rec / fpr > max_rec_fpr:
            max_rec_fpr = rec / fpr
            best_threshold = threshold
        print('(Threshold={:.1f}): Loss: {:.4f}, Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, FPR: {:.4f}, TP: {}, FP: {}, FN: {}, TN: {}'.format(
            threshold, val_loss, acc, prec, rec, fpr, tp, fp, fn, tn))

    print('Setting threshold to %.2f' % best_threshold)
    # global thresh
    thresh = best_threshold
    valF.write('{},{},{},{},{},{}\n'.format(epoch, best_threshold, val_loss, acc, prec, rec))
    valF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    # correct = 0
    acc = prec = rec = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.binary_cross_entropy(output, target).data[0]
        pred = output.data.gt(thresh)
        # correct += pred.eq(target.data.byte()).cpu().sum()
        tp = (pred + target.data.byte()).eq(2).sum()
        fp = (pred - target.data.byte()).eq(1).sum()
        fn = (pred - target.data.byte()).eq(-1).sum()
        tn = (pred + target.data.byte()).eq(0).sum()
        # correct = pred.eq(target.data.byte()).cpu().sum()
        # acc = 100. * correct / (pred.size()[0] * pred.size()[1])
        acc += (tp + tn) / (tp + tn + fp + fn)
        try:
            prec += tp / (tp + fp)
        except ZeroDivisionError:
            prec += 0.0
        try:
            rec += tp / (tp + fn)
        except ZeroDivisionError:
            rec += 0.0

    test_loss /= len(testLoader) # loss function already averages over batch size
    # nTotal = len(testLoader.dataset) * 11
    # acc = 100. * correct/nTotal
    acc /= len(testLoader)
    prec /= len(testLoader)
    rec /= len(testLoader)
    print('\nTest set: Loss: {:.4f}, Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}\n'.format(
        test_loss, acc, prec, rec))

    testF.write('{},{},{},{},{}\n'.format(epoch, test_loss, acc, prec, rec))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 20: lr = 1e-1
        elif epoch == 30: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
