import torch.utils.data as data
import os
import json
import torch
import numpy as np
import random
import math
from .transformations import get_transform, get_transform_strong_Weak


def _parse_slice_filename(filename):
    basename = filename.replace(".npz", "")
    parts = basename.split("_slice_")
    case_name = parts[0].replace("vol_", "", 1)
    slice_name = parts[1]
    return case_name, slice_name


class ProstateDataset(data.Dataset):
    """Dataset for PROSTATE .npz slices with metadata.json-driven splits."""

    def __init__(self, data_root, domain_name, phase="train", split_train=True,
                 img_size=(384, 384), weak_strong_aug=False):
        self.data_root = data_root
        self.domain_name = domain_name
        self.phase = phase
        self.img_size = img_size
        self.weak_strong_aug = weak_strong_aug

        if self.weak_strong_aug:
            self.augmenter_w, self.augmenter_s = get_transform_strong_Weak(
                self.phase, New_size=img_size
            )
        else:
            self.augmenter = get_transform(self.phase, New_size=img_size)

        metadata_path = os.path.join(data_root, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

        self.slice_dir = os.path.join(data_root, domain_name, "slices")
        split_key = "train" if split_train else "test"
        self.all_data_path = []
        self.name_list = []

        if self.metadata and "splits" in self.metadata:
            case_ids = set(self.metadata["splits"][domain_name][split_key])
            for f in sorted(os.listdir(self.slice_dir)):
                if not f.endswith(".npz"):
                    continue
                case_name, _ = _parse_slice_filename(f)
                if case_name in case_ids:
                    self.all_data_path.append(os.path.join(self.slice_dir, f))
                    self.name_list.append(f.replace(".npz", ""))
        else:
            for f in sorted(os.listdir(self.slice_dir)):
                if not f.endswith(".npz"):
                    continue
                self.all_data_path.append(os.path.join(self.slice_dir, f))
                self.name_list.append(f.replace(".npz", ""))

    def __len__(self):
        return len(self.all_data_path)

    def __getitem__(self, index):
        raw = np.load(self.all_data_path[index])
        name = self.name_list[index]

        img = raw["img"].astype(np.float32)
        img -= img.min()
        img /= img.max() + 1e-8
        img = np.repeat(img[None, ...], 3, axis=0).transpose((1, 2, 0))  # (H, W, 3)

        seg = raw["label"].astype(np.int64)  # (H, W), values 0/1

        if self.weak_strong_aug:
            transformed_w = self.augmenter_w(image=img, mask=seg)
            img_w = transformed_w["image"]
            seg = transformed_w["mask"]
            transformed_s = self.augmenter_s(
                image=img_w.numpy().transpose((1, 2, 0))
            )
            img_s = transformed_s["image"]
            return img_w, img_s, seg, name
        else:
            transformed = self.augmenter(image=img, mask=seg)
            img = transformed["image"]
            img = img.to(torch.float32)
            seg = transformed["mask"]
            seg = seg.to(torch.long)
            return img, seg, name


class ProstatePatientDataset(data.Dataset):
    """Patient-level dataset for PROSTATE with metadata.json-driven splits."""

    def __init__(self, data_root, domain_name, phase="train", split_train=True,
                 img_size=(384, 384)):
        self.data_root = data_root
        self.phase = phase
        self.augmenter = get_transform(self.phase, New_size=img_size)

        metadata_path = os.path.join(data_root, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None

        slice_dir = os.path.join(data_root, domain_name, "slices")
        split_key = "train" if split_train else "test"

        if self.metadata and "splits" in self.metadata:
            case_ids = set(self.metadata["splits"][domain_name][split_key])
        else:
            case_ids = None

        self.all_data_path = []
        self.name_list = []
        self.patients = []
        self.patients_slices = {}

        file_list = sorted(os.listdir(slice_dir))
        case_to_paths = {}
        for f in file_list:
            if not f.endswith(".npz"):
                continue
            case_name, _ = _parse_slice_filename(f)
            if case_ids is not None and case_name not in case_ids:
                continue
            case_to_paths.setdefault(case_name, []).append(
                os.path.join(slice_dir, f)
            )

        start = 0
        for case_name in sorted(case_to_paths.keys()):
            paths = case_to_paths[case_name]
            self.patients.append(case_name)
            for p in paths:
                self.all_data_path.append(p)
                self.name_list.append(os.path.basename(p).replace(".npz", ""))
            end = len(self.all_data_path)
            self.patients_slices[case_name] = [start, end]
            start = end

    def __len__(self):
        return len(self.all_data_path)

    def __getitem__(self, index):
        raw = np.load(self.all_data_path[index])
        name = self.name_list[index]

        img = raw["img"].astype(np.float32)
        img -= img.min()
        img /= img.max() + 1e-8
        img = np.repeat(img[None, ...], 3, axis=0).transpose((1, 2, 0))

        seg = raw["label"].astype(np.int64)

        transformed = self.augmenter(image=img, mask=seg)
        img = transformed["image"].to(torch.float32)
        seg = transformed["mask"].to(torch.long)
        return img, seg, name
