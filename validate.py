import torch
from dataloader import MyDataset
from torch.utils.data import Dataset, DataLoader
from utilities.Transforms import test_transform
from model import classifier, classifier_sep
import argparse

if __name__ == '__main__':
    # Command parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--checkpoint', type=str, required=True, help="model checkpoint")
    args = parser.parse_args()
    
    # Dataloader
    valid_data = MyDataset('./dataset/metadata/dev.csv', 'dev', transform=test_transform)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False, num_workers=8)

    clf = classifier()
    clf.load_state_dict(torch.load(args.checkpoint))
    clf = clf.cuda()
    clf.eval()
    
    correct = 0
    preds = []
    ans = []
    for i, (batch_input, target) in enumerate(valid_loader):
        batch_input = batch_input.cuda()
        target = target.cuda()
        output, _ = clf(batch_input)

        pred = torch.max(output, 1)[1]
        correct += (pred == target).sum()
        preds.extend(pred.tolist())
        ans.extend(target.tolist())

    acc = correct.item() / len(valid_loader.dataset)
    print ("Acc {:.3f} ({}/{})".format(acc, correct, len(valid_loader.dataset)))
    print ("-------------------------")

    class_wise = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    num = [0, 0, 0]

    for i in range(len(preds)):
        a = ans[i]
        p = preds[i]
        class_wise[a][p] += 1
        num[a] += 1

    for i in range(3):
        print ("Class {}: ".format(i+1), end='')
        for j in range(3):
            print ("{}/{} ".format(class_wise[i][j], num[i]), end='')
        print ('')
                
    

