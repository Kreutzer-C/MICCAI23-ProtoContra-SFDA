# Proto_Contra_SFDA
## Source-Free Domain Adaptation for Medical Image Segmentation via Prototype-Anchored Feature Alignment and Contrastive Learning

This is an official implementation of MICCAI 2023 paper [Source-Free Domain Adaptation for Medical Image Segmentation via Prototype-Anchored Feature Alignment and Contrastive Learning.](https://arxiv.org/abs/2307.09769)

![](images/pipeline.png)

### Requirements
- Linux with Python >= **3.7**
- PyTorch >= **1.7.1** and [torchvision](https://github.com/pytorch/vision/) matches the PyTorch installation. Install them following the official instructions from [pytorch.org](https://pytorch.org) to make sure of this.
- Additional dependencies: `einops`, `surface_distance`, `albumentations`, `tensorboardX`, `tqdm`, `pyyaml`, `scipy`

### Supported Datasets

| Dataset | Classes | Image Size | Format | Domains |
|---------|---------|-----------|--------|---------|
| **CHAOS** | 5 (bg + 4 organs) | 256x256 | .npy stacked | CT / MR |
| **PROSTATE** | 2 (bg + prostate) | 384x384 | .npz (`img`/`label`) | source (BMC+RUNMC), target_1 (BIDMC+HK+UCL), target_2 (I2CVB) |

#### PROSTATE Dataset Structure
```
datasets/PROSTATE/processed_new/
  metadata.json              # train/test splits by case ID per domain
  source/slices/             # 805 .npz files
  target_1/slices/           # 594 .npz files
  target_2/slices/           # 468 .npz files
```
Each `.npz` file contains:
- `img`: float32 array (384, 384), normalized to [0, 1]
- `label`: uint8 array (384, 384), binary mask (0/1)

### Quick Start

##### 1. Clone repository
```bash
git clone https://github.com/CSCYQJ/MICCAI23-ProtoContra-SFDA
cd MICCAI23-ProtoContra-SFDA
```

##### 2. Download Data
- **CHAOS**: CT data from [MICCAI 2015 Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789). MRI data from [2019 CHAOS Challenge](https://chaos.grand-challenge.org/).
- **PROSTATE**: Multi-site prostate MRI dataset. Place processed data under `datasets/PROSTATE/processed_new/` with `metadata.json` defining train/test splits.

---

### CHAOS Pipeline (Original)

##### 3. Source Model Training
```bash
python main_trainer_source.py --config_file configs/train_source_seg.yaml --gpu_id 0
```

##### 4. Target Domain Adaptation - PFA Stage
```bash
python main_trainer_sfda.py --config_file configs/train_target_adapt_PFA.yaml --gpu_id 0
```

##### 5. Target Domain Adaptation - CL Stage
```bash
python main_trainer_sfda.py --config_file configs/train_target_adapt_CL.yaml --gpu_id 0
```

---

### PROSTATE Pipeline

#### Step 1: Source Domain Pre-training

Train a UNet segmentation model on the source domain (BMC + RUNMC) with CE + Dice loss:

```bash
python main_trainer_source.py \
    --config_file configs/train_prostate_source_seg.yaml \
    --gpu_id 0
```

Key config (`configs/train_prostate_source_seg.yaml`):
- `dataset: "PROSTATE"`, `num_classes: 2`, `img_size: [384, 384]`
- `source_domain: "source"`, `total_epochs: 100`, `lr: 0.001`

#### Step 2: Test Source Pre-trained Model

Evaluate the source model on all domains to establish baselines:

```bash
# Test on source domain (in-domain performance)
python test.py \
    --dataset PROSTATE \
    --domain source \
    --data_root datasets/PROSTATE/processed_new \
    --model_path results/Source_Seg/<exp_dir>/saved_models/best_model_xxx.pth \
    --gpu_id 0

# Test on target_1 (cross-domain)
python test.py \
    --dataset PROSTATE \
    --domain target_1 \
    --data_root datasets/PROSTATE/processed_new \
    --model_path results/Source_Seg/<exp_dir>/saved_models/best_model_xxx.pth \
    --gpu_id 0

# Test on target_2 (cross-domain)
python test.py \
    --dataset PROSTATE \
    --domain target_2 \
    --data_root datasets/PROSTATE/processed_new \
    --model_path results/Source_Seg/<exp_dir>/saved_models/best_model_xxx.pth \
    --gpu_id 0
```

Test results (per-patient Dice, ASSD, and summary) are automatically saved as `test_results_<domain>.txt` in the experiment folder (parent of `saved_models/`).

#### Step 3: Target Domain Adaptation - PFA Stage

Update `source_model_path` in `configs/train_prostate_target_adapt_PFA.yaml` to point to the best source model, then run:

```bash
python main_trainer_sfda.py \
    --config_file configs/train_prostate_target_adapt_PFA.yaml \
    --gpu_id 0
```

Key config (`configs/train_prostate_target_adapt_PFA.yaml`):
- `target_domain: "target_1"`, `total_epochs: 5`, `lr: 0.0001`
- Uses PCT bidirectional transport loss to align target features to source prototypes

#### Step 4: Target Domain Adaptation - CL Stage

Update `source_model_path` in `configs/train_prostate_target_adapt_CL.yaml` to point to the best PFA model, then run:

```bash
python main_trainer_sfda.py \
    --config_file configs/train_prostate_target_adapt_CL.yaml \
    --gpu_id 0
```

Key config (`configs/train_prostate_target_adapt_CL.yaml`):
- `target_domain: "target_1"`, `total_epochs: 5`, `lr: 0.0001`
- `low_rank: 1`, `high_rank: 2` (adapted for 2-class segmentation)
- `num_queries: 64`, `num_negatives: 256`

#### Step 5: Test Adapted Model

```bash
python test.py \
    --dataset PROSTATE \
    --domain target_1 \
    --data_root datasets/PROSTATE/processed_new \
    --model_path results/Target_Adapt/<CL_exp_dir>/saved_models/best_model_xxx.pth \
    --gpu_id 0
```

---

### Test Script (`test.py`)

Unified testing script supporting both CHAOS and PROSTATE datasets.

**Features**:
- 3D volume-level Dice and ASSD metric computation (per-patient)
- Per-patient breakdown and per-class summary printed to console
- Results automatically saved as `test_results_<domain>.txt` in the model's experiment folder

**Arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `CHAOS` | Dataset type (`CHAOS` or `PROSTATE`) |
| `--model_path` | (required) | Path to model checkpoint `.pth` |
| `--data_root` | `datasets/chaos` | Dataset root directory |
| `--domain` | `source` | PROSTATE domain (`source`, `target_1`, `target_2`) |
| `--target_site` | `MR` | CHAOS target site |
| `--num_classes` | auto | Number of classes (auto: 5 for CHAOS, 2 for PROSTATE) |
| `--img_size` | auto | Image size (auto: 256 for CHAOS, 384 for PROSTATE) |
| `--gpu_id` | `0` | GPU device ID |
| `--batch_size` | `16` | Batch size for inference |

**Output file example** (`test_results_target_1.txt`):
```
Test Results
Time: 2026-06-21 18:08:15
Dataset: PROSTATE
Domain: target_1
Model: best_model_step_70_dice_0.8037.pth
...

Patient              Prostate Dice Prostate ASSD     Avg Dice     Avg ASSD
--------------------------------------------------------------------------
BIDMC_Case02               0.7697       ...          0.7697       ...
...

Class                 Dice       ASSD
-------------------------------------
Prostate            0.8037     4.1565
-------------------------------------
Mean (fg)           0.8037     4.1565
=====================================
```

---

### Full Pipeline Script (PROSTATE, one-click)

Below is a complete shell script to run the entire pipeline:

```bash
#!/bin/bash
# Full ProtoContra pipeline for PROSTATE dataset
# Usage: bash run_prostate_pipeline.sh <gpu_id>

GPU_ID=${1:-0}
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)
DATA_ROOT="${PROJECT_DIR}/datasets/PROSTATE/processed_new"

# ---- Step 1: Source Pre-training ----
echo "==== Step 1: Source Domain Pre-training ===="
python main_trainer_source.py \
    --config_file configs/train_prostate_source_seg.yaml \
    --gpu_id ${GPU_ID}

# Find best source model
SOURCE_BEST=$(find results/Source_Seg/UNet_Prostate_Source_Seg -name "best_model_*.pth" | sort -t_ -k5 -rn | head -1)
echo "Best source model: ${SOURCE_BEST}"

# ---- Step 1.5: Test Source Model ----
echo "==== Step 1.5: Test Source Model ===="
for domain in source target_1 target_2; do
    python test.py --dataset PROSTATE --domain ${domain} \
        --data_root ${DATA_ROOT} --model_path "${SOURCE_BEST}" --gpu_id ${GPU_ID}
done

# ---- Step 2: PFA Stage ----
echo "==== Step 2: PFA Adaptation ===="
sed -i "s|^source_model_path:.*|source_model_path: '${SOURCE_BEST}'|" \
    configs/train_prostate_target_adapt_PFA.yaml
python main_trainer_sfda.py \
    --config_file configs/train_prostate_target_adapt_PFA.yaml \
    --gpu_id ${GPU_ID}

# Find best PFA model
PFA_BEST=$(find results/Target_Adapt/UNet_Prostate_Source2Target1_Adapt_PFA -name "best_model_*.pth" | sort -t_ -k5 -rn | head -1)
echo "Best PFA model: ${PFA_BEST}"

# ---- Step 3: CL Stage ----
echo "==== Step 3: CL Adaptation ===="
sed -i "s|^source_model_path:.*|source_model_path: '${PFA_BEST}'|" \
    configs/train_prostate_target_adapt_CL.yaml
python main_trainer_sfda.py \
    --config_file configs/train_prostate_target_adapt_CL.yaml \
    --gpu_id ${GPU_ID}

# Find best CL model
CL_BEST=$(find results/Target_Adapt/UNet_Prostate_Source2Target1_Adapt_CL -name "best_model_*.pth" | sort -t_ -k5 -rn | head -1)
echo "Best CL model: ${CL_BEST}"

# ---- Step 4: Test Adapted Model ----
echo "==== Step 4: Test Adapted Model ===="
for domain in target_1 target_2; do
    python test.py --dataset PROSTATE --domain ${domain} \
        --data_root ${DATA_ROOT} --model_path "${CL_BEST}" --gpu_id ${GPU_ID}
done

echo "==== Pipeline Complete ===="
```

---

### Project Structure

```
MICCAI23-ProtoContra-SFDA/
  main_trainer_source.py       # Source domain training entry
  main_trainer_sfda.py         # Target adaptation entry (PFA / CL)
  test.py                      # Unified test script (CHAOS + PROSTATE)
  configs/
    train_source_seg.yaml                  # CHAOS source training
    train_target_adapt_PFA.yaml            # CHAOS PFA adaptation
    train_target_adapt_CL.yaml             # CHAOS CL adaptation
    train_prostate_source_seg.yaml         # PROSTATE source training
    train_prostate_target_adapt_PFA.yaml   # PROSTATE PFA adaptation
    train_prostate_target_adapt_CL.yaml    # PROSTATE CL adaptation
  dataloaders/
    dataloaders.py             # CHAOS dataset (MyDataset, PatientDataset)
    prostate_dataloader.py     # PROSTATE dataset (ProstateDataset, ProstatePatientDataset)
  trainers/
    source_seg_trainer.py      # Source supervised trainer
    target_adapt_PFA_trainer.py    # PFA adaptation trainer
    target_adapt_CL_trainer.py     # CL adaptation trainer
    target_adapt_pseudo_label_trainer.py
  models/
    unet.py                    # UNet architecture
  losses/                      # ProtoLoss, PseudoLabel_Loss, etc.
  utils/
    metrics.py                 # MultiDiceScore, MultiASD (3D volume-level)
```

### Citation
If you find this work or code is helpful in your research, please cite:
```
@article{yu2023source,
  title={Source-Free Domain Adaptation for Medical Image Segmentation via Prototype-Anchored Feature Alignment and Contrastive Learning},
  author={Yu, Qinji and Xi, Nan and Yuan, Junsong and Zhou, Ziyu and Dang, Kang and Ding, Xiaowei},
  journal={arXiv preprint arXiv:2307.09769},
  year={2023}
}
```

### Acknowledgement

Many thanks to these excellent opensource projects 
* [PCT](https://github.com/korawat-tanwisuth/Proto_DA) 
* [U2PL](https://haochen-wang409.github.io/U2PL)
