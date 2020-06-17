import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from utilities.Transforms import transform, test_transform
from dataloader import MyDataset
from model import classifier, classifier_sep
import numpy as np
import sys

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(c_epoch, clf, train_loader, optimizer):
    print ("Epoch {}: ".format(c_epoch), end='')

    running_loss = 0
    for i, (batch_input, target) in enumerate(train_loader):
        batch_input = batch_input.cuda()
        target = target.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        # inputs, targets_a, targets_b, lam = mixup_data(batch_input, target)
        input_var = torch.autograd.Variable(batch_input)
        # target_a_var = torch.autograd.Variable(targets_a)
        # target_b_var = torch.autograd.Variable(targets_b)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, _ = clf(input_var)
        # loss = mixup_criterion(criterion, output, target_a_var, target_b_var, lam)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print ("Loss {:.3f}, ".format(running_loss), end='')
        
def validate(c_epoch, clf, valid_loader):
    correct = 0
    for i, (batch_input, target) in enumerate(valid_loader):
        batch_input = batch_input.cuda()
        target = target.cuda()
        output, _ = clf(batch_input)

        pred = torch.max(output, 1)[1]
        correct += (pred == target).sum()
    

    acc = correct.item() / len(valid_loader.dataset)
    print ("Acc {:.3f}".format(acc))
    return acc
    
if __name__ == '__main__':
    # Dataloader
    train_data = MyDataset('./dataset/metadata/train.csv', 'train', transform=transform)
    valid_data = MyDataset('./dataset/metadata/dev.csv', 'dev', transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False, num_workers=8)

    # Model
    clf = classifier().cuda()
    print (clf)

    # Optimizer
    optimizer = optim.SGD(clf.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    # Train
    patience = 0
    current_best = 0
    epochs = 50
    for epoch in range(epochs):
        clf.train()
        train(epoch, clf, train_loader, optimizer)
        
        # Validation
        acc = validate(epoch, clf, valid_loader)
        
        # Early stop
        if acc <= current_best:
            patience += 1
        else:
            print ("Saving best model...")
            torch.save(clf.state_dict(), './checkpoints/alex_vgg_clf_crop.pt')
            current_best = acc
            patience = 0
        if patience > 10:
            break

        scheduler.step()
    

