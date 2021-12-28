import argparse
from time import gmtime, strftime
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from dataloader import Superbin

from arch.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from arch.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from efficientnet_pytorch import EfficientNet
import train
import val
import wandb
import cv2

if __name__ == '__main__':
    wandb.init(project='superbin')
    wandb.run.name = 'test'
    wandb.run.save()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='vgg19_bn', choices=['vgg19_bn'])
    parser.add_argument('--lr_base', type=float, default=0.0125)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr_drop_epochs', type=int, default=[10, 20, 30], nargs='+')
    parser.add_argument('--lr_drop_rate', type=float, default=0.1)
    args = parser.parse_args()
    wandb.config.update(args)
    
   # model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=4)
    # define model
    
    if args.arch.startswith('resnet'):
        if args.arch == 'resnet18':
            model = resnet18(num_classes=4)
        elif args.arch == 'resnet34':
            model = resnet34(num_classes=4)
        elif args.arch == 'resnet50':
            model = resnet50(num_classes=4)
        elif args.arch == 'resnet101':
            model = resnet101(num_classes=4)
        elif args.arch == 'resnet152':
            model = resnet152(num_classes=4)
        else:
            raise NotImplementedError(f"architecture {args.arch} is not implemented")
    elif args.arch.startswith('vgg'):
        if args.arch == 'vgg11_bn':
            model = vgg11_bn()
        elif args.arch == 'vgg13_bn':
            model = vgg13_bn()
        elif args.arch == 'vgg16_bn':
            model = vgg16_bn()
        elif args.arch == 'vgg19_bn':
            model = vgg19_bn()
        else:
            raise NotImplementedError(f"architecture {args.arch} is not implemented")
    else:
        raise NotImplementedError(f"architecture {args.arch} is not implemented")
    
    model = model.cuda()
    model = torch.nn.parallel.DataParallel(model)
    wandb.watch(model)
    
    transform_train = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    transform_val = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

   
    dataloader_train = DataLoader(Superbin(1, transform_train), shuffle=True, num_workers=10, batch_size=args.batch_size)
    dataloader_val = DataLoader(Superbin(0, transform_val), shuffle=False, num_workers=10, batch_size=args.batch_size)


    # LR schedule
    lr = args.lr_base
    lr_per_epoch = []
    for epoch in range(args.epochs):
        if epoch in args.lr_drop_epochs:
            lr *= args.lr_drop_rate
        lr_per_epoch.append(lr)

    # define loss and optimizer(손실 함수와 옵티마이저 SGD)
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)

    # save_path
    current_time = strftime('%Y-%m-%d_%H:%M', gmtime())
    save_dir = os.path.join(f'checkpoints/{current_time}')
    os.makedirs(save_dir,  exist_ok=True)

    # train and val
    best_perform, best_epoch = -100, -100
    for epoch in range(1, args.epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_per_epoch[epoch-1]
        print(f"Training at epoch {epoch}. LR {lr_per_epoch[epoch-1]}")

        train.train(model, dataloader_train, criterion, optimizer, epoch=epoch)
        val.val(model, dataloader_val, epoch=epoch)

        save_data = {'epoch': epoch,
         #            'acc': acc,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        
        torch.save(save_data, os.path.join(save_dir, f'{epoch:03d}.pth.tar'))
        if epoch > 1:
            os.remove(os.path.join(save_dir, f'{epoch-1:03d}.pth.tar'))
    #    if acc >= best_perform:
    #        torch.save(save_data, os.path.join(save_dir, 'best.pth.tar'))
    #        best_perform = acc1
    #        best_epoch = epoch
    #    print(f"best performance {best_perform} at epoch {best_epoch}")
    #    wandb.log({
    #    "val_acc": acc
    #})
