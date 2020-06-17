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
    test_data = MyDataset('./dataset/metadata/test.csv', 'test', transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=8)

    clf = classifier()
    clf.load_state_dict(torch.load(args.checkpoint))
    clf = clf.cuda()
    clf.eval()
    
    f = open('test.csv', 'w')

    preds = []
    for i, (batch_input, target) in enumerate(test_loader):
        batch_input = batch_input.cuda()
        target = target.cuda()
        output, _ = clf(batch_input)

        pred = torch.max(output, 1)[1]
        preds.extend(pred.tolist())

    fh = open('./dataset/metadata/test.csv')
    i = 0
    f.write("image_id,label\n")
    for line in fh.readlines()[1:]:
        line = line.strip('\n')
        f.write(line)
        if preds[i] == 0:
            pred = 'A'
        elif preds[i] == 1:
            pred = 'B'
        else:
            pred = 'C'
        f.write(pred)
        f.write('\n')
        i += 1

    

