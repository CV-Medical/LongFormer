import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model



def get_args_parser():
    parser = argparse.ArgumentParser('Longformer', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr_drop', default=40, type=int, nargs='+')


    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='unet3d', type=str,
                        help="Name of the convolutional backbone to use: [unet3d,]")
    parser.add_argument('--num_feature_scales', default=4, type=int, help='number of feature levels/scales')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=288, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--nheads', default=6, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_classes', default=4, type=int,
                        help="Number of instance_classes, 5/36")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)


    # * Matcher
    parser.add_argument('--set_cost_loc', default=5, type=float, help="Localization coefficient in the matching cost")
    parser.add_argument('--set_cost_cls', default=2, type=float, help="Classification coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--loc_loss_coef', default=5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='ADNI')
    parser.add_argument('--classification_type', default='NC/AD', help='NC/AD or sMCI/pMCI')
    parser.add_argument('--data_path', default='/data/qiuhui/data/adni')
    parser.add_argument('--num_visits', default=1, type=int, help='number of visits')
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--model', default=None, help='load from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
        
    return parser


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    
    model.to(device)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
        sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    
    
    if isinstance(model, nn.Module):
        param_groups = model.parameters()
    else:
        param_groups = model
    optimizer = torch.optim.Adam(
            param_groups,
            lr=5e-4,
            eps=1e-4,
            weight_decay=0.0,
        )
    

    output_dir = Path(args.output_dir)


    if not args.eval: 
        print("Start training")
        start_time = time.time()
        for epoch in range(args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)
            train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch)

            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if (epoch + 1) % 1 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            
            evaluate(model, 
                    criterion,
                    data_loader_val, 
                    device, 
                    args)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    else:
        print("Start validation")
        if args.model is not None:
            print('load from ',args.model)
            checkpoint = torch.load(args.model, map_location='cpu')
            model.load_state_dict(checkpoint['model'],strict=True)
        else:
            raise ValueError('please provide model by "--model"')
        evaluate(model, 
                criterion,
                data_loader_val, 
                device, 
                args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Longformer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



