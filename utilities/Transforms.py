import torch
import torchvision.transforms as transforms

def transform(x, y):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    t_x = t(x)
    
    if y == 'A':
        t_y = torch.tensor(0)
    elif y == 'B':
        t_y = torch.tensor(1)
    else:
        t_y = torch.tensor(2)

    return t_x, t_y.long()

def mixup_transform(x, y):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    t_x = t(x)
    
    if y == 'A':
        t_y = torch.tensor(0)
    elif y == 'B':
        t_y = torch.tensor(1)
    else:
        t_y = torch.tensor(2)

    return t_x, t_y.long()

def test_transform(x, y):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    t = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    t_x = t(x)
    
    if y == 'A':
        t_y = torch.tensor(0)
    elif y == 'B':
        t_y = torch.tensor(1)
    else:
        t_y = torch.tensor(2)

    return t_x, t_y.long()
