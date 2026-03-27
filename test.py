"""
Test script for MICCAI23-ProtoContra-SFDA.

Usage:
    python3 test.py \
        --model_path results/Target_Adapt/.../saved_models/best_model.pth \
        --data_root datasets/chaos \
        --target_site MR \
        --gpu_id 0
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from dataloaders import MyDataset
from utils.metrics import MultiDiceScore, MultiASD


# Actual label order from preprocess_chaos.ipynb: select_label()
# liver=1, R.Kidney=2, L.Kidney=3, Spleen=4
# Note: the yaml organ_list has Liver/Spleen swapped (a labeling bug, not affecting metric values)
ORGAN_LIST = ['Liver', 'R.Kidney', 'L.Kidney', 'Spleen']
NUM_CLASSES = 5


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint (.pth)')
    parser.add_argument('--data_root', type=str,
                        default='datasets/chaos',
                        help='Root directory of the dataset')
    parser.add_argument('--target_site', type=str, default='MR',
                        help='Target domain site name (e.g. MR, CT)')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--arch', type=str, default='UNet')
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def build_model(args, device):
    cfg = {
        'arch': args.arch,
        'input_dim': args.input_dim,
        'num_classes': args.num_classes,
    }
    model = get_model(cfg)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {args.model_path}')
    return model


def collect_predictions(model, dataloader, device):
    """Run inference and group slice predictions by patient."""
    sample_dict = {}
    with torch.no_grad():
        for images, segs, names in tqdm(dataloader, desc='Inference'):
            images = images.to(device)
            _, predicts = model(images)
            for i, name in enumerate(names):
                parts = name.split('_')
                patient_id = parts[0]
                slice_idx = int(parts[1])
                entry = (predicts[i].cpu(), segs[i].cpu(), slice_idx)
                sample_dict.setdefault(patient_id, []).append(entry)
    return sample_dict


def build_volumes(sample_dict):
    """Sort slices per patient and stack into 3D volumes, skip all-zero slices."""
    pred_volumes, gt_volumes = [], []
    for patient_id in sorted(sample_dict.keys()):
        slices = sorted(sample_dict[patient_id], key=lambda x: x[2])
        preds, targets = [], []
        for pred, target, _ in slices:
            if target.sum() == 0:
                continue
            preds.append(pred)
            targets.append(target)
        if len(preds) == 0:
            continue
        pred_volumes.append(torch.stack(preds, dim=-1))    # (C, H, W, D)
        gt_volumes.append(torch.stack(targets, dim=-1))    # (H, W, D)
    return pred_volumes, gt_volumes


def compute_metrics(pred_volumes, gt_volumes, num_classes, organ_list):
    """Compute per-class Dice and ASSD for all patients."""
    num_fg = num_classes - 1
    all_dice = np.full((len(pred_volumes), num_fg), np.nan)
    all_assd = np.full((len(pred_volumes), num_fg), np.nan)

    for idx, (pred, gt) in enumerate(zip(pred_volumes, gt_volumes)):
        dice_list = MultiDiceScore(pred, gt, num_classes, include_bg=False)
        for c, d in enumerate(dice_list):
            if not np.isnan(d):
                all_dice[idx, c] = d

        try:
            assd_list = MultiASD(pred, gt, num_classes, include_bg=False)
            for c, a in enumerate(assd_list):
                all_assd[idx, c] = a
        except Exception:
            pass

    results = {}
    print('\n' + '=' * 60)
    print(f'{"Class":<15} {"Dice":>10} {"ASSD":>10}')
    print('-' * 60)
    for c, organ in enumerate(organ_list):
        dice_mean = np.nanmean(all_dice[:, c])
        assd_mean = np.nanmean(all_assd[:, c])
        results[organ] = {'dice': dice_mean, 'assd': assd_mean}
        print(f'{organ:<15} {dice_mean:>10.4f} {assd_mean:>10.4f}')

    mean_dice = np.nanmean(all_dice)
    mean_assd = np.nanmean(all_assd)
    results['mean'] = {'dice': mean_dice, 'assd': mean_assd}
    print('-' * 60)
    print(f'{"Mean (fg)":<15} {mean_dice:>10.4f} {mean_assd:>10.4f}')
    print('=' * 60)
    return results


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Target site: {args.target_site}')

    dataset = MyDataset(
        rootdir=args.data_root,
        sites=[args.target_site],
        phase='val',
        split_train=False,
    )
    print(f'Test dataset size: {len(dataset)}')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = build_model(args, device)
    sample_dict = collect_predictions(model, dataloader, device)
    pred_volumes, gt_volumes = build_volumes(sample_dict)
    print(f'Total patients evaluated: {len(pred_volumes)}')

    organ_list = ORGAN_LIST[:args.num_classes - 1]
    compute_metrics(pred_volumes, gt_volumes, args.num_classes, organ_list)


if __name__ == '__main__':
    main()
