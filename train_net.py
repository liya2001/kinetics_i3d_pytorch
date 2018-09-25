import argparse
import os
import time
import shutil
import pandas as pd
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from src.i3dpt import I3D
from opts import parser
from transforms import *
from dataset import ViratDataSet
from config import LABEL_MAPPING_2_CLASS, LABEL_MAPPING_3_CLASS, LABEL_MAPPING_2_CLASS2

best_prec1 = 0


def get_augmentation(modality, input_size):
    if modality == 'RGB':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                               GroupRandomHorizontalFlip(is_flow=False)])
    elif modality == 'Flow':
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75]),
                                               GroupRandomHorizontalFlip(is_flow=True)])


def compose_transform(mode, modality):
    crop_size = 224
    scale_size = 256
    # ToDo: is it right for Flow?
    if modality == 'Flow':
        input_mean = [0.5]
        input_std = np.mean([0.229, 0.224, 0.225])
    else:
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]

    # policies = model.get_optim_policies()
    # ToDo: augmentation like i3d or tsn?
    train_augmentation = get_augmentation(modality, crop_size)
    normalize = GroupNormalize(input_mean, input_std)

    if mode == 'train':
        transform = torchvision.transforms.Compose([
            train_augmentation,
            # ToDo: roll, div
            Stack(roll=True),
            ToTorchFormatTensor(div=False),
            normalize,
        ])
    else:
        transform = torchvision.transforms.Compose([
            GroupScale(int(scale_size)),
            GroupCenterCrop(crop_size),
            Stack(roll=True),
            ToTorchFormatTensor(div=False),
            normalize,
        ])

    return transform


def main():
    global args, best_prec1
    args = parser.parse_args()
    print('Called with args:')
    print(args)

    if args.mapping == 'mapping_2':
        MAPPING_LABEL = LABEL_MAPPING_2_CLASS
    elif args.mapping == 'mapping_3':
        MAPPING_LABEL = LABEL_MAPPING_3_CLASS
    else:
        MAPPING_LABEL = None

    model = I3D(num_classes=args.num_classes)

    # load pretrained kinetics model
    if args.pretrained_weights:
        model.load_state_dict(torch.load(args.pretrained_weights))

    # ToDo:
    param_count = 0
    for n, param in model.named_parameters():
        param_count += 1
        # freeze first four layer
        # n.startswith('base_model.inception_4') or n.startswith('base_model.inception_5') \
        if len(n) > 21 and n[21] in args.freeze_layers:
            param.requires_grad = False
        print(param_count, n, param.requires_grad, param.size())

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # ToDo: consistent with I3D
    if args.modality == 'RGB':
        data_length = 1
    else:  # args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    weights = make_weights_for_unbalance_classes(args.train_list_file, mapping=MAPPING_LABEL)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.Tensor(weights), weights.size)
    train_loader = torch.utils.data.DataLoader(
        ViratDataSet(args.data_path, args.train_list_file,
                     new_length=data_length,
                     modality=args.modality,
                     transform=compose_transform('train', args.modality),
                     reverse=args.reverse,
                     mapping=MAPPING_LABEL),
        batch_size=args.batch_size, shuffle=False, sampler=sampler,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ViratDataSet(args.data_path, args.val_list_file,
                     new_length=data_length,
                     modality=args.modality,
                     transform=compose_transform('val', args.modality),
                     test_mode=True,
                     mapping=MAPPING_LABEL),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")
    # ToDo: policies
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, epoch+1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top2.update(prec2.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top2=top2, lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec2 = accuracy(output.data, target, topk=(1, 2))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top2.update(prec2.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top2=top2)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} (Best:{best_prec1:.3f}) Prec@2 {top2.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, best_prec1=best_prec1, top2=top2, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, str(epoch), args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def make_weights_for_unbalance_classes(label_csv_path, mapping):
    dir_label_df = pd.read_csv(label_csv_path, sep=' ', header=None)
    sample_num = dir_label_df.shape[0]
    print('All samples num:', sample_num)

    dir_label_df.columns = ['activity', 'length', 'label', 'offset', 'reverse', 'mapping']
    if mapping:
        dir_label_df['label'] = dir_label_df['label'].map(lambda x: mapping[x])
    weights = np.zeros(sample_num)
    num_class = 1 + dir_label_df['label'].max()
    for i in range(num_class):
        this_class_sample = dir_label_df[dir_label_df['label'] == i]
        assert this_class_sample.shape[0] != 0
        weight = sample_num / this_class_sample.shape[0]
        if i == 13:
            weight = weights.min()+1.
        weights[dir_label_df['label'] == i] = weight
        print('class', i, this_class_sample.shape[0], weight)

    return weights


if __name__ == '__main__':
    main()
