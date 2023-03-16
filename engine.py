import os
from typing import Iterable
import cv2
import numpy as np


import torch
from datasets.data_prefetcher import data_prefetcher

from sklearn import metrics
from sklearn.preprocessing import label_binarize



cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def train_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int):
    model.train()
    criterion.train()

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    idx = 0

    for _ in range(len(data_loader)):

        outputs, loss_dict, indices_list = model(samples, targets, criterion, train=True)
        print(outputs['pred_labels'].argmax(dim=-1)) 
        print('pred: ', outputs['pred_labels'][:,indices_list[-1][0][0]]) 
        print('pred: ', outputs['pred_labels'].argmax(dim=-1)[:,indices_list[-1][0][0]]) 
        print('gt: ', targets[0]['label'])

        print(idx, loss_dict)
        idx+=1

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        
        optimizer.step()
   
        samples, targets = prefetcher.next()
     
    torch.cuda.empty_cache()



@torch.no_grad()
def evaluate(model, criterion, data_loader, device, args):


    model.eval()
    criterion.eval()

    y_gt = []
    y_pred = []
    index = 0

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in range(len(data_loader)):
        print('{}/{}'.format(index,len(data_loader)))
        index+=1

        samples = samples.to(device)

        outputs, loss_dict, indices_list = model(samples, targets, criterion, train=False)
        
        print(outputs['pred_labels'].argmax(dim=-1)) 
        print('pred: ', outputs['pred_labels'][:,indices_list[-1][0][0]]) 
        print('pred: ', outputs['pred_labels'].argmax(dim=-1)[:,indices_list[-1][0][0]]) 
        print('gt: ', targets[0]['label'])

        bs = outputs['pred_labels'].shape[0]

        for b_idx in range(bs):
            pred_i, gt_j = indices_list[-1][b_idx]

            pred_label = outputs['pred_labels'][b_idx][pred_i].argmax(dim=1)
            gt_label = targets[b_idx]['label'][gt_j]

            print('pred: ', pred_label[0], '\ngt: ', gt_label[0])
            
            y_gt.append(int(gt_label[0]))
            y_pred.append(int(pred_label[0]))

        samples, targets = prefetcher.next()
    
    
    acc = metrics.accuracy_score(y_gt, y_pred)
    print('acc: '+str(acc))
    f1 = metrics.f1_score(y_gt, y_pred,average='micro')
    print('f1: '+str(f1))

    class_names = np.unique(y_gt)
    fpr, tpr, _= metrics.roc_curve(label_binarize(y_gt, classes=class_names).ravel(),label_binarize(y_pred, classes = class_names).ravel())
    auc = metrics.auc(fpr, tpr)
    print('auc: '+str(auc))

    with open(os.path.join(args.output_dir, 'log.txt'), 'a+') as f:
        f.write('accuracy: {}, AUC: {}\n'.format(acc,auc))
    return

