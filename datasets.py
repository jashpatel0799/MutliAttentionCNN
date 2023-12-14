import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor

def build_transform(args,is_train=True):
    if args.image_size==224:
        src_size=256
    elif args.image_size==448:
        src_size=480

    if is_train:
        transform = transforms.Compose([
            transforms.Resize(src_size),
            transforms.RandomCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(src_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return transform

def build_dataloader(args):
    transform_train = build_transform(args, is_train=True)
    # dataset_train=ImageFolder(os.path.join(args.dataset_path,"train"),
    #                           transform_train)
    # dataloader_train=DataLoader(dataset_train,batch_size=args.batch_size,
    #                             shuffle=True,num_workers=args.nthreads,drop_last=True)

    # transform_val = build_transform(args, is_train=False)
    # dataset_val = ImageFolder(os.path.join(args.dataset_path, "val"),
    #                           transform_val)
    # dataloader_val = DataLoader(dataset_val, batch_size=8,
    #                               shuffle=False, num_workers=args.nthreads)
    dataset = ImageFolder(os.path.join(args.dataset_path,"images"),
                              transform_train)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    dataset_train, dataset_val = random_split(dataset, [train_size, test_size])
    batch_size = 16
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    # print(type(dataset))
    num_classes=len(dataset.classes)
    n_iter_per_epoch_train=len(dataloader_train)


    return dataloader_train,dataloader_val,dataset_train,dataset_val,\
           num_classes,n_iter_per_epoch_train

